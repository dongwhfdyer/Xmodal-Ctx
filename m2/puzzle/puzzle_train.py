# Create the dataset
import pickle


from data import ImageDetectionsField, TextField, RawField, DataLoader
from data.field import puzzleIdField
from puzzle.puzzle_model import puzzleSolver
from puzzle_opt import p_opt
from data.dataset import PuzzleCOCO
# import Path
from pathlib import Path


def prepareField():
    object_field = ImageDetectionsField(obj_file=Path(p_opt.dataset_root) / p_opt.obj_file, max_detections=50, preload=p_opt.preload)
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    vocab_file = 'vocab/vocab_coco.pkl'
    text_field.vocab = pickle.load(open(vocab_file, 'rb'))
    puzzle_field = puzzleIdField(puzzleFile=datasetRoot / p_opt.puzzle_file, )
    return object_field, text_field, puzzle_field


def genOneItem(dataloader):
    data = dataloader.__iter__().__next__()
    obj = data["object"].to(cudaDevice, non_blocking=True)
    puzzle = data["puzzle_id"].to(cudaDevice, non_blocking=True)  # kuhn: what's non_blocking?
    captions = data["text"].to(cudaDevice, non_blocking=True)
    return obj, puzzle, captions


def build_model():
    return puzzleSolver().to(cudaDevice)


if __name__ == '__main__':
    cudaDevice = "cuda:0"
    # Create the dataset
    datasetRoot = Path(p_opt.dataset_root)
    object_field, text_field, puzzle_field = prepareField()

    fields = {
        "object": object_field,
        "text": text_field,
        "img_id": RawField(),
        "puzzle_id": puzzle_field,
    }
    dset = datasetRoot / "annotations"
    puzzlecoco = PuzzleCOCO(fields, dset, dset)

    train_dataset, val_dataset, test_dataset = puzzlecoco.splits
    train_dataloader = DataLoader(train_dataset, batch_size=p_opt.batch_size, shuffle=True, num_workers=0, drop_last=True)
    model = build_model()
    # ---------kkuhn-block------------------------------ # kuhn: only for testing

    # ad = train_dataset.__getitem__(0)
    obj, puzzle, captions = genOneItem(train_dataloader)
    out = model(obj=obj, puzzle=puzzle, caption=captions)
    dd = train_dataloader.__iter__().__next__()

    # ---------kkuhn-block------------------------------

    print("--------------------------------------------------")
