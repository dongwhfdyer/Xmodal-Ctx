import argparse
from pathlib import Path
import h5py
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import Trainer, LightningModule, seed_everything
from transformers import CLIPModel, CLIPProcessor

import sys

from distributed_utils import init_distributed_mode, setup_for_distributed
from utils import load_huggingface_model

sys.path.append('.')
from dataset import CocoImageCrops, collate_crops


class ImageEncoder(LightningModule):
    def __init__(self, save_dir):
        super().__init__()

        CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
        LOCAL_CLIP_FILE = "pretrained/clip-vit-base-patch32.pt"
        self.save_dir = Path(save_dir)
        self.model = load_huggingface_model(CLIPModel, CLIP_MODEL_NAME, LOCAL_CLIP_FILE, return_vision_model=True)

    def test_step(self, batch, batch_idx):
        orig_imgs, _, _, _, ids = batch

        features = self.model(pixel_values=orig_imgs)
        features = features.pooler_output
        features = features.detach().cpu().numpy()

        with h5py.File(self.save_dir / "vis_ctx.hdf5", "a") as f:
            f.attrs["fdim"] = features.shape[-1]
            for i in range(len(orig_imgs)):
                f.create_dataset(str(int(ids[i])), data=features[i])


def func1(args):
    return torch.FloatTensor(args["pixel_values"][0])


def build_ctx_caps(args):
    CLIP_PROCESSOR_NAME = "openai/clip-vit-base-patch32"
    LOCAL_CLIPPROCESSOR_FILE = "pretrained/clip-processor.pt"
    clip_processor = load_huggingface_model(CLIPProcessor, CLIP_PROCESSOR_NAME, LOCAL_CLIPPROCESSOR_FILE, return_feature_extractor=True)
    transform = T.Compose([
        clip_processor,
        func1,
        #        lambda x: torch.FloatTensor(x["pixel_values"][0]),
    ])
    dset = CocoImageCrops(args.dataset_root / "annotations", args.dataset_root, transform)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(dset)

    dloader = DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_crops,
        # sampler=val_sampler,
        pin_memory=True,

    )

    img_encoder = ImageEncoder(args.save_dir)
    # img_encoder = torch.nn.parallel.DistributedDataParallel(img_encoder, device_ids=[args.device], find_unused_parameters=True)

    trainer = Trainer(
        accelerator="ddp",
        gpus=args.device,
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir,
    )
    trainer.test(img_encoder, dloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode images')
    parser.add_argument('--exp_name', type=str, default='temp')
    # parser.add_argument('--exp_name', type=str, default='image_features')
    parser.add_argument('--dataset_root', type=str, default='datasets/coco_captions')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--device', default=[0, 1], help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("outputs") / args.exp_name)
    shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(1, workers=True)

    build_ctx_caps(args)
