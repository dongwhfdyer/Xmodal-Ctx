# Create the dataset
import os
import pickle
import shutil
from pathlib import Path

from torch.nn import NLLLoss

from data import ImageDetectionsField, TextField, TxtCtxField, VisCtxField, RawField, COCO
from models import Transformer
from models.transformer import MemoryAugmentedEncoder, ScaledDotProductAttentionMemory, MeshedDecoder, Projector
from original_args import args
from data.quickerdataset import COCO
from data import DataLoader


def prepareArgs():
    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("outputs") / args.exp_name)
    if not (args.resume_last or args.resume_best):
        shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)


def buildModel():
    encoder = MemoryAugmentedEncoder(
        3, 0, attention_module=ScaledDotProductAttentionMemory,
        attention_module_kwargs={'m': args.m}
    )
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    projector = Projector(
        f_obj=2054, f_vis=vis_ctx_field.fdim, f_txt=516,  # kuhn edited
        f_out=encoder.d_model, drop_rate=args.drop_rate
    )

    model = Transformer(
        bos_idx=text_field.vocab.stoi['<bos>'],
        encoder=encoder, decoder=decoder, projector=projector
    ).to(cudaDevice)
    return model


cudaDevice = "cuda:1"


def genOneItem(dataloader):
    data = dataloader.__iter__().__next__()
    txt_ctx = {
        k1: {
            k2: v2.to(cudaDevice, non_blocking=True)
            for k2, v2 in v1.items()
        }
        for k1, v1 in data["txt_ctx"].items()
    }
    vis_ctx = data["vis_ctx"].to(cudaDevice, non_blocking=True)
    obj = data["object"].to(cudaDevice, non_blocking=True)
    captions = data["text"].to(cudaDevice, non_blocking=True)
    return obj, vis_ctx, txt_ctx, captions


def prepareField():
    object_field = ImageDetectionsField(obj_file=args.dataset_root / args.obj_file, max_detections=50, preload=args.preload)
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    vocab_file = 'vocab/vocab_coco.pkl'
    text_field.vocab = pickle.load(open(vocab_file, 'rb'))
    txt_ctx_field = TxtCtxField(ctx_file=args.dataset_root / "txt_ctx.hdf5", k=args.topk, preload=args.preload)
    vis_ctx_field = VisCtxField(ctx_file=args.dataset_root / "vis_ctx.hdf5", preload=args.preload)
    return object_field, text_field, text_field, txt_ctx_field, vis_ctx_field


if __name__ == '__main__':
    prepareArgs()

    object_field, text_field, text_field, txt_ctx_field, vis_ctx_field = prepareField()

    fields = {
        "object": object_field, "text": text_field, "img_id": RawField(),
        "txt_ctx": txt_ctx_field, "vis_ctx": vis_ctx_field,
    }

    dset = args.dataset_root / "annotations"
    cocodataset = COCO(fields, dset, dset)
    dataset = cocodataset.image_dictionary()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>']).to(cudaDevice)

    model = buildModel()
    obj, vis_ctx, txt_ctx, captions = genOneItem(dataloader)
    out = model(obj=obj, vis_ctx=vis_ctx, txt_ctx=txt_ctx, seq=captions, mode="xe")
    out = out[:, :-1].contiguous()
    captions_gt = captions[:, 1:].contiguous()
    loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

    pass
