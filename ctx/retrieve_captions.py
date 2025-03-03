import argparse
import json
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import math
import faiss
from pytorch_lightning.distributed import dist
from tqdm import tqdm
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import Trainer, LightningModule, seed_everything
from transformers import CLIPModel, CLIPProcessor

import sys

sys.path.append('.')
from dataset import CocoImageCrops, collate_crops, CocoImage, collate_no_crops, gqaImage, CocoImage_for_mdetr, gqaImage_for_mdetr, flickrImage_for_mdetr, Image_for_mdetr, dataset_crop, collate_crops_v2, MyImageFolder


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
            if filenames[i] in self.image_caption_pairs:
                raise ValueError(f"Duplicate filename {filenames[i]}")
            self.image_caption_pairs[filenames[i]] = [self.text[j] for j in I_o[i]]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        orig_imgs, five_imgs, nine_imgs, filenames = batch
        N = len(orig_imgs)

        with h5py.File(self.save_dir / "txt_ctx_{}.hdf5".format(self.global_rank), "a") as f:
            D_o, I_o = self.search(orig_imgs, topk=self.k)  # N x self.k

            D_f, I_f = self.search(torch.flatten(five_imgs, end_dim=1), topk=self.k)  # N*5 x self.k
            D_f, I_f = D_f.reshape(N, 5, self.k), I_f.reshape(N, 5, self.k)

            D_n, I_n = self.search(torch.flatten(nine_imgs, end_dim=1), topk=self.k)  # N*9 x self.k
            D_n, I_n = D_n.reshape(N, 9, self.k), I_n.reshape(N, 9, self.k)

            for i in range(N):
                g1 = f.create_group(filenames[i])

                texts = [self.text[j] for j in I_o[i]]
                features = self.features[I_o[i]]
                g2 = g1.create_group("whole")
                g2.create_dataset("features", data=features)
                g2.create_dataset("texts", data=texts)

                texts = [
                    [
                        self.text[I_f[i, j, k]]
                        for k in range(self.k)
                    ]
                    for j in range(5)
                ]
                features = self.features[I_f[i].flatten()].reshape((5, self.k, -1))
                g3 = g1.create_group("five")
                g3.create_dataset("features", data=features)
                g3.create_dataset("texts", data=texts)

                texts = [
                    [
                        self.text[I_n[i, j, k]]
                        for k in range(self.k)
                    ]
                    for j in range(9)
                ]
                features = self.features[I_n[i].flatten()].reshape((9, self.k, -1))
                g4 = g1.create_group("nine")
                g4.create_dataset("features", data=features)
                g4.create_dataset("texts", data=texts)

    # def on_predict_end(self) -> None:
    #     # wait all processes to finish and merge the results
    #     if self.global_rank == 0:
    #         print("Merging results")
    #         with h5py.File(self.save_dir / "txt_ctx.hdf5", "w") as f:
    #             for i in range(self.global_rank, self.world_size):
    #                 with h5py.File(self.save_dir / "txt_ctx_{}.hdf5".format(i), "r") as f1:
    #                     for k in f1.keys():
    #                         f.copy(f1[k], k)
    #         print("Done")


def generate_dataset(args):
    transform = T.Compose([
        CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").feature_extractor,
        lambda x: torch.FloatTensor(x["pixel_values"][0]),
    ])
    if "gqa" in args.exp_name:
        dset = Image_for_mdetr(args.dataset_root, args.ann_dir, transform, "gqa")

    elif "coco" in args.exp_name:
        dset = Image_for_mdetr(args.dataset_root, args.ann_dir, transform, "coco")

    elif "flickr" in args.exp_name:
        dset = Image_for_mdetr(args.dataset_root, args.ann_dir, transform, "flickr")
    else:
        raise Exception("no such datasets for {}".format(args.exp_name))

    # if "gqa" in args.exp_name:
    #     dset = gqaImage_for_mdetr(args.dataset_root, args.ann_dir, transform=transform)
    # elif "coco" in args.exp_name:
    #     dset = CocoImage_for_mdetr(args.dataset_root, args.ann_dir, transform=transform)
    # elif "flickr" in args.exp_name:
    #     dset = flickrImage_for_mdetr(args.dataset_root, args.ann_dir, transform=transform)
    # else:
    #     raise Exception("no such datasets for {}".format(args.exp_name))

    return dset


def generate_dataset_for_image_captioning(args):
    transform = T.Compose([
        CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").feature_extractor,
        lambda x: torch.FloatTensor(x["pixel_values"][0]),
    ])
    # dset = dataset_crop(args.dataset_root, args.ann_dir, transform, "coco")  # todo: it's for coco2014
    dset = MyImageFolder(args.dataset_root, transform)  # todo: it's for coco2017. All the coco2017 images is put under the same folder

    return dset


def build_ctx_caps(args):
    # dset = generate_dataset(args) # todo
    dset = generate_dataset_for_image_captioning(args)
    dloader = DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_crops_v2 if 'crop' in args.exp_name else collate_no_crops
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
    # trainer.test(cap_retr, dloader)
    trainer.predict(cap_retr, dloader)

    # save to json
    import json
    # wait for all processes to finish
    # with open(args.save_dir / "image_caption_pairs.json", "w") as f:
    #     json.dump(cap_retr.image_caption_pairs, f)

    aa = {}
    for i, (k, v) in tqdm(enumerate(cap_retr.image_caption_pairs.items())):
        aa[k] = v

    with open(args.save_dir / "image_caption_pairs.json", "w") as f:
        json.dump(aa, f)

    # ---------kkuhn-block------------------------------ # read from json
    with open(args.save_dir / "image_caption_pairs.json", "r") as f:
        dd = json.load(f)
    # ---------kkuhn-block------------------------------


def add_two_json_together():
    json1 = Path("outputs/retrieved_captions_coco_100/image_caption_pairs.json")
    json2 = Path("outputs/retrieved_captions_gqa_100/image_caption_pairs.json")

    with open(json1, 'r') as f:
        data1 = json.load(f)
    with open(json2, 'r') as f:
        data2 = json.load(f)

    data1.update(data2)

    with open("outputs/retrieved_captions_coco_100/image_caption_pairs_all.json", 'w') as f:
        json.dump(data1, f, indent=4)
    # read json
    with open("outputs/retrieved_captions_coco_100/image_caption_pairs_all.json", 'r') as f:
        data = json.load(f)


def test_data_integrity():
    final_mixed_train_json = Path("ctx/datasets/coco_captions/annotations/OpenSource/final_mixed_train.json")
    with open(final_mixed_train_json, "r") as f:
        json_content = json.load(f)
    len(json_content["images"])
    data = []
    for image_info in json_content["images"]:
        data.append(image_info["file_name"])
    # find unique image ids
    data = list(set(data))

    with open("outputs/retrieved_captions_coco_100/image_caption_pairs.json", 'r') as f:
        pairs_all = json.load(f)
    len(pairs_all)

    # with open("outputs/retrieved_captions_coco_100/image_caption_pairs_all.json", 'r') as f:
    #     pairs_all = json.load(f)

    n = 0
    for filename in data:
        if filename[:-4] not in pairs_all:
            print(filename[:-4])
            n += 1
    print(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve captions')
    parser.add_argument('--device', type=int, default=[4, 5], nargs='+', help='GPU device')
    parser.add_argument('--ann_dir', type=str, default='ctx/datasets/coco_captions/annotations/OpenSource')
    # parser.add_argument('--exp_name', type=str, default='retrieved_captions_flickr_100')  # todo: must be set
    # parser.add_argument('--dataset_root', type=str, default='ctx/datasets/flickr30k-images')
    # parser.add_argument('--exp_name', type=str, default='retrieved_captions_gqa_100')  # todo: must be set
    # parser.add_argument('--dataset_root', type=str, default='/home/szh2/datasets/gqa/images')  # todo: must be set
    # parser.add_argument('--exp_name', type=str, default='retrieved_captions_coco_100')  # todo: must be set
    # parser.add_argument('--dataset_root', type=str, default='datasets/coco_captions')
    # parser.add_argument('--exp_name', type=str, default='coco_crop_test')  # todo: must be set
    # parser.add_argument('--dataset_root', type=str, default='/home/szh2/datasets/coco/val2014')
    # parser.add_argument('--exp_name', type=str, default='coco_crop_test')  # todo: must be set
    # parser.add_argument('--dataset_root', type=str, default='/home/szh2/datasets/coco/test2014')
    parser.add_argument('--exp_name', type=str, default='coco2017all_crop')  # todo: must be set
    parser.add_argument('--dataset_root', type=str, default='/home/szh2/kuhn/smallcap/data/images')

    parser.add_argument('--caption_db', type=str, default='ctx/outputs/captions_db/caption_db.hdf5')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=10)
    args = parser.parse_args()

    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("outputs") / args.exp_name)
    shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(args)

    seed_everything(1, workers=True)

    build_ctx_caps(args)
