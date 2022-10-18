# Create the dataset
from data import ImageDetectionsField, TextField, RawField
from data.field import puzzleIdField
from puzzle_opt import p_opt
from data.dataset import PuzzleCOCO
# import Path
from pathlib import Path

if __name__ == '__main__':
    # Create the dataset
    datasetRoot = Path(p_opt.dataset_root)
    object_field = ImageDetectionsField(
        obj_file=datasetRoot / p_opt.obj_file,
        max_detections=50, preload=p_opt.preload
    )
    text_field = TextField(
        init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
        remove_punctuation=True, nopoints=False
    )
    puzzle_field = puzzleIdField(
        puzzleFile=datasetRoot / p_opt.puzzle_file,
    )
    fields = {
        "object": object_field,
        "text": text_field,
        "img_id": RawField(),
        "puzzle_id": puzzle_field,
    }
    dset = datasetRoot / "annotations"
    puzzlecoco = PuzzleCOCO(fields, dset, dset)

    train_dataset, val_dataset, test_dataset = puzzlecoco.splits

    print("--------------------------------------------------")

