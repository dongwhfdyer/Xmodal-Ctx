import argparse
from pathlib import Path
import h5py
import numpy as np
import math
import faiss
from tqdm import tqdm
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import Trainer, LightningModule, seed_everything
from transformers import CLIPModel, CLIPProcessor

import sys

sys.path.append('.')
from dataset import CocoImageCrops, collate_crops, CocoImage, collate_no_crops, gqaImage


class CaptionRetriever(LightningModule):
    def __init__(self, caption_db, save_dir, k):
        super().__init__()

        self.save_dir = Path(save_dir)
        self.k = k

        self.keys, self.features, self.text = self.load_caption_db(caption_db)
        self.index = self.build_index(idx_file=self.save_dir / "faiss.index")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_caption_pairs = {}

    @staticmethod
    def load_caption_db(caption_db):
        print("Loading caption db")
        keys, features, text = [], [], []  # through text_projection, features are turned into keys
        with h5py.File(caption_db, "r") as f:
            for i in tqdm(range(len(f))):
                keys_i = f[f"{i}/keys"][:]
                features_i = f[f"{i}/features"][:]
                text_i = [str(x, "utf-8") for x in f[f"{i}/captions"][:]]

                keys.append(keys_i)
                features.append(features_i)
                text.extend(text_i)
        keys = np.concatenate(keys)
        features = np.concatenate(features)

        return keys, features, text

    def build_index(self, idx_file):
        print("Building db index")
        n, d = self.keys.shape
        K = round(8 * math.sqrt(n))
        # index_factory: preprocess, quantizer, metric_type
        index = faiss.index_factory(d, f"IVF{K},Flat", faiss.METRIC_INNER_PRODUCT)  # faiss.METRIC_INNER_PRODUCT means cosine similarity
        assert not index.is_trained
        index.train(self.keys)  # train on the dataset, to set the centroids of the k-means
        assert index.is_trained
        index.add(self.keys)  # add vectors to the index
        index.nprobe = max(1, K // 10)  # nprobe is the number of clusters to search

        faiss.write_index(index, str(idx_file))

        return index

    def search(self, images, topk):
        features = self.clip.vision_model(pixel_values=images)[1]  # pooler_output is the last hidden state of the [CLS] token (bs, 768)
        query = self.clip.visual_projection(features)  # (bs, 512)
        query = query / query.norm(dim=-1, keepdim=True)
        D, I = self.index.search(query.detach().cpu().numpy(), topk)

        return D, I

    def test_step(self, batch, batch_idx):
        orig_imgs, filenames = batch
        N = len(orig_imgs)
        for i in range(N):
            D_o, I_o = self.search(orig_imgs, topk=self.k)  # D_o(distance): (query_N, topk), I_o(index): (query_N, topk)
            self.image_caption_pairs[filenames[i]] = [self.text[j] for j in I_o[i]]


def build_ctx_caps(args):
    transform = T.Compose([
        CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").feature_extractor,
        lambda x: torch.FloatTensor(x["pixel_values"][0]),
    ])
    if "gqa" in args.exp_name:
        dset = gqaImage(args.dataset_root, transform=transform)
    elif "coco" in args.exp_name:
        dset = CocoImage(args.dataset_root / "annotations", args.dataset_root, transform)
    dloader = DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_no_crops
    )

    cap_retr = CaptionRetriever(
        caption_db=args.caption_db,
        save_dir=args.save_dir,
        k=args.k
    )

    trainer = Trainer(
        # fast_dev_run=True,
        gpus=args.device,
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir,
        strategy="ddp"
    )
    trainer.test(cap_retr, dloader)
    # save to json
    import json
    with open(args.save_dir / "image_caption_pairs.json", "w") as f:
        json.dump(cap_retr.image_caption_pairs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve captions')
    parser.add_argument('--device', type=int, default=[0, 1, 2, 3], nargs='+', help='GPU device')
    parser.add_argument('--exp_name', type=str, default='retrieved_captions_gqa_100')  # todo: must be set
    parser.add_argument('--dataset_root', type=str, default='/home/szh2/datasets/gqa/images')  # todo: must be set
    # parser.add_argument('--dataset_root', type=str, default='datasets/coco_captions')
    parser.add_argument('--caption_db', type=str, default='ctx/outputs/captions_db/caption_db.hdf5')
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()

    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("outputs") / args.exp_name)
    shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(args)

    seed_everything(1, workers=True)

    build_ctx_caps(args)
