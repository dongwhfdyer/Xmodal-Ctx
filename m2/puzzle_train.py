from pathlib import Path

from data import TextField, TxtCtxField, RawField, COCO
from m2.data import ImageDetectionsField, VisCtxField
from puzzle_opt import p_opt

if __name__ == '__main__':
    dataset_root = Path(p_opt.dataset_root)
    # Create the dataset
    object_field = ImageDetectionsField(
        obj_file=Path(dataset_root) / p_opt.obj_file,
        max_detections=50, preload=p_opt.preload
    )
    text_field = TextField(
        init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
        remove_punctuation=True, nopoints=False
    )
    txt_ctx_filed = TxtCtxField(
        ctx_file=dataset_root / "txt_ctx.hdf5", k=p_opt.topk, preload=p_opt.preload
    )
    vis_ctx_filed = VisCtxField(
        ctx_file=dataset_root / "vis_ctx.hdf5", preload=p_opt.preload
    )
    fields = {
        "object": object_field, "text": text_field, "img_id": RawField(),
        "txt_ctx": txt_ctx_filed, "vis_ctx": vis_ctx_filed
    }
    dset = dataset_root / "annotations"
    dataset = COCO(fields, dset, dset)
    # each dataset has many examples
    # example = {
    #     "img_id": img_id,
    #     "object": img_id,
    #     "text": caption,
    #     "txt_ctx": img_id,
    #     "vis_ctx": img_id
    # }
    train_dataset, val_dataset, test_dataset = dataset.splits
