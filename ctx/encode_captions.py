import argparse
from pathlib import Path
import shutil
import h5py
import numpy as np

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule, seed_everything
from transformers import CLIPModel

import sys

from utils import load_huggingface_model

sys.path.append('.')
from dataset import VisualGenomeCaptions, collate_tokens

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LOCAL_CLIP_FILE = "pretrained/clip-vit-base-patch32.pt"


class CaptionDB(LightningModule):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        # self.clip: projection_dim_text_and_vision=512, text_embed_dim=512, visual_embed_dim=768
        # self.clip.text_model: 512 -> 512
        # self.clip.text_projection: 512 -> 512
        # self.clip.visual_projection: 768 -> 512
        self.clip = load_huggingface_model(CLIPModel, CLIP_MODEL_NAME, LOCAL_CLIP_FILE)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None
        # captions: (batch_size, max_len)
        # tokens: input_ids: (batch_size, max_len), attention_mask: (batch_size, max_len)
        captions, tokens = batch
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        # self.clip.text_model(**tokens): last_hidden_state (bs, max_len, 512), pooler_output (bs, 512)
        # So, It's extracting the pooler_output
        features = self.clip.text_model(**tokens)[1]
        keys = self.clip.text_projection(features)
        keys = keys / keys.norm(dim=-1, keepdim=True)

        features = features.detach().cpu().numpy()
        keys = keys.detach().cpu().numpy()

        # captions(list): bs
        # features(ndarray): (bs, embed_dim)
        # keys(ndarray): (bs, text_projection_dim)
        with h5py.File(self.save_dir / "caption_db.hdf5", "a") as f:
            g = f.create_group(str(batch_idx))
            g.create_dataset("keys", data=keys, compression="gzip")
            g.create_dataset("features", data=features, compression="gzip")
            g.create_dataset("captions", data=captions, compression="gzip")


def build_caption_db(args):
    dset = VisualGenomeCaptions(args.ann_dir)
    dloader = DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_tokens
    )
    cap_db = CaptionDB(args.save_dir)

    trainer = Trainer(
        gpus=args.device ,
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir

    )
    trainer.test(cap_db, dloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode captions')
    parser.add_argument('--device', type=int, default=[0,1,2,3], nargs='+', help='GPU device(s) to use')
    parser.add_argument('--exp_name', type=str, default='captions_db')
    parser.add_argument('--ann_dir', type=str, default='datasets/visual_genome')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    setattr(args, "save_dir", Path("outputs") / args.exp_name)
    shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(args)

    seed_everything(1, workers=True)

    build_caption_db(args)
