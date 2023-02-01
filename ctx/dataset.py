from PIL import Image
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO as pyCOCO
import json
import itertools

from torchvision.datasets import ImageFolder
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from torchvision.transforms import functional as F

from utils import load_huggingface_model

LENGTH_LIMIT = 75


def collate_tokens(batch):
    captions, input_ids, attention_mask, lengths = [], [], [], []
    for cap, tok in batch:
        assert tok["input_ids"].shape == tok["attention_mask"].shape
        captions.append(cap)

        l = tok["input_ids"].shape[1]
        if l < LENGTH_LIMIT:
            input_ids.append(tok["input_ids"])
            attention_mask.append(tok["attention_mask"])
            lengths.append(l)
        else:
            input_ids.append(tok["input_ids"][:, :LENGTH_LIMIT])
            attention_mask.append(tok["attention_mask"][:, :LENGTH_LIMIT])
            lengths.append(LENGTH_LIMIT)

    max_len = max(lengths)
    input_pad, atten_pad = [], []
    for i in range(len(input_ids)):
        l = input_ids[i].shape[1]
        if l < max_len:
            p = torch.zeros(size=(1, max_len - l), dtype=input_ids[i].dtype)
            input_pad.append(torch.cat([input_ids[i], p], dim=1))

            p = torch.zeros(size=(1, max_len - l), dtype=attention_mask[i].dtype)
            atten_pad.append(torch.cat([attention_mask[i], p], dim=1))
        else:
            input_pad.append(input_ids[i])
            atten_pad.append(attention_mask[i])

    input_pad = torch.cat(input_pad)
    atten_pad = torch.cat(atten_pad)
    assert input_pad.shape[1] <= LENGTH_LIMIT
    assert atten_pad.shape[1] <= LENGTH_LIMIT
    assert input_pad.shape == atten_pad.shape

    tokens = {"input_ids": input_pad, "attention_mask": atten_pad}

    return captions, tokens


class VisualGenomeCaptions(Dataset):
    def __init__(self, ann_dir):
        super().__init__()
        CLIPPROCESSOR_NAME = "openai/clip-vit-base-patch32"
        LOCAL_CLIPPROCESSOR_FILE = "pretrained/clip-processor.pt"
        LOCAL_CAPS_FILE = "cache/captions.txt"
        self.tokenizer = load_huggingface_model(CLIPProcessor, CLIPPROCESSOR_NAME, LOCAL_CLIPPROCESSOR_FILE, return_tokenizer=True)
        escapes = ''.join([chr(char) for char in range(0, 32)])
        self.translator = str.maketrans('', '', escapes)

        self.caps = self.read_caps_cache_or_process(LOCAL_CAPS_FILE, ann_dir)

    def read_caps_cache_or_process(self, LOCAL_CAPS_FILE, ann_dir):
        if Path(LOCAL_CAPS_FILE).exists():
            print("loading captions from cache...")
            with open(LOCAL_CAPS_FILE, "r") as f:
                self.caps = f.read().splitlines()
        else:
            print("parsing captions...")
            self.caps = self.parse_annotations_v2(Path(ann_dir))
            with open(LOCAL_CAPS_FILE, "w") as f:
                f.write("\n".join(self.caps))
            print("saved captions to cache...")
        return self.caps

    @staticmethod
    def combination(l1, l2):
        return [" ".join(x) for x in itertools.product(l1, l2)]

    def process_word(self, s):
        return s.lower().strip().translate(self.translator)

    def process_synset(self, s):
        return s.lower().strip().translate(self.translator).split(".")[0]

    def parse_annotations(self, ann_dir):
        print("loading object attributes...")
        objs = {}
        with open(ann_dir / "attributes.json", "r") as f:
            attributes = json.load(f)
        for x in tqdm(attributes, dynamic_ncols=True):
            for a in x["attributes"]:
                _names = set(self.process_synset(y) for y in a.get("synsets", list()))
                _attrs = set(self.process_word(y) for y in a.get("attributes", list()))

                for n in _names:
                    try:
                        objs[n] |= _attrs
                    except KeyError:
                        objs[n] = _attrs
        del attributes

        print("loading object relationships...")
        rels = set()
        with open(ann_dir / "relationships.json", "r") as f:
            relationships = json.load(f)
        for x in tqdm(relationships, dynamic_ncols=True):
            for r in x["relationships"]:
                _pred = self.process_word(r["predicate"])
                _subj = set(self.process_synset(y) for y in r["subject"]["synsets"])
                _obj = set(self.process_synset(y) for y in r["object"]["synsets"])

                for s in _subj:
                    for o in _obj:
                        rels.add(f"{s}<sep>{_pred}<sep>{o}")
        # cache object relationships

        del relationships

        print("parsing object attributes...")
        caps_obj = []
        for o in tqdm(objs.keys()):
            for a in objs[o]:
                if a != "":  # skip empty attributes
                    caps_obj.append(f"{a} {o}")  # attribute + object

        print("parsing object relationships...")
        caps_rel = []
        for r in tqdm(rels):
            s, p, o = r.split("<sep>")
            caps_rel.append(f"{s} {p} {o}")  # subject + predicate + object

        caps = np.unique(caps_obj + caps_rel).tolist()
        return caps

    def parse_annotations_v2(self, ann_dir):
        print("loading object attributes...")
        objs = {}
        with open(ann_dir / "attributes.json", "r") as f:
            attributes = json.load(f)
        for x in tqdm(attributes, dynamic_ncols=True):
            for a in x["attributes"]:
                _names = set(self.process_synset(y) for y in a.get("synsets", list()))
                _attrs = set(self.process_word(y) for y in a.get("attributes", list()))

                for n in _names:
                    try:
                        objs[n] |= _attrs
                    except KeyError:
                        objs[n] = _attrs
        del attributes

        print("parsing object attributes...")
        caps_obj = []
        for o in tqdm(objs.keys()):
            for a in objs[o]:
                if a != "":  # skip empty attributes
                    caps_obj.append(f"{a} {o}")  # attribute + object

        caps = np.unique(caps_obj).tolist()
        return caps

    def __len__(self):
        return len(self.caps)

    def __getitem__(self, index):
        tokens = self.tokenizer(self.caps[index], padding=True, return_tensors="pt")

        return self.caps[index], tokens


def collate_crops(data):
    orig_image, five_images, nine_images, captions, idx = zip(*data)

    orig_image = torch.stack(list(orig_image), dim=0)
    five_images = torch.stack(list(five_images), dim=0)
    nine_images = torch.stack(list(nine_images), dim=0)
    captions = list(captions)
    idx = torch.LongTensor(list(idx))

    return orig_image, five_images, nine_images, captions, idx


def collate_no_crops(data):
    orig_image, filename = zip(*data)

    orig_image = torch.stack(list(orig_image), dim=0)
    filename = list(filename)

    return orig_image, filename


class CocoImageCrops(Dataset):
    def __init__(self, ann_dir, img_root, transform=None):
        self.transform = transform
        self.data = self.parse(Path(ann_dir), Path(img_root))

    @staticmethod
    def parse(ann_dir, img_root):
        ids = (
            np.load(ann_dir / "coco_train_ids.npy"),
            np.concatenate([
                np.load(ann_dir / "coco_restval_ids.npy"),
                np.load(ann_dir / "coco_dev_ids.npy"),
                np.load(ann_dir / "coco_test_ids.npy")
            ]),
        )
        coco = (
            pyCOCO(ann_dir / "captions_train2014.json"),
            pyCOCO(ann_dir / "captions_val2014.json"),
        )
        img_root = (img_root / "train2014", img_root / "val2014")

        data = {}
        for i in range(len(ids)):
            for idx in ids[i]:
                img_id = coco[i].anns[idx]["image_id"]
                img_file = img_root[i] / coco[i].loadImgs(img_id)[0]["file_name"]
                caption = coco[i].anns[idx]["caption"].strip()

                if img_id in data:  # one image has multiple captions
                    data[img_id]["captions"].append(caption)
                else:
                    data[img_id] = {
                        "image_id": img_id,
                        "image_file": img_file,
                        "captions": [caption, ]
                    }

        data = list(data.values())
        data.sort(key=lambda x: x["image_id"])

        return data

    def five_crop(self, image, ratio=0.6):
        w, h = image.size
        hw = (h * ratio, w * ratio)

        return F.five_crop(image, hw)

    def nine_crop(self, image, ratio=0.4):
        w, h = image.size

        t = (0, int((0.5 - ratio / 2) * h), int((1.0 - ratio) * h))
        b = (int(ratio * h), int((0.5 + ratio / 2) * h), h)
        l = (0, int((0.5 - ratio / 2) * w), int((1.0 - ratio) * w))
        r = (int(ratio * w), int((0.5 + ratio / 2) * w), w)
        h, w = list(zip(t, b)), list(zip(l, r))

        images = []
        for s in itertools.product(h, w):
            h, w = s
            top, left = h[0], w[0]
            height, width = h[1] - h[0], w[1] - w[0]
            images.append(F.crop(image, top, left, height, width))

        return images

    def trapezoid(self, image, ratio=0):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index]["image_file"])
        image = image.convert("RGB")

        five_images = self.five_crop(image)
        nine_images = self.nine_crop(image)
        # trapezoid_images = self.trapezoid(image)

        if self.transform is not None:
            orig_image = self.transform(image)
            five_images = torch.stack([self.transform(x) for x in five_images])
            nine_images = torch.stack([self.transform(x) for x in nine_images])

        captions = self.data[index]["captions"]
        idx = self.data[index]["image_id"]
        # orig_image: (3, 224, 224)
        # five_images: (5, 3, 224, 224)
        # nine_images: (9, 3, 224, 224)
        # captions: list of str
        # idx: int (image id)
        return orig_image, five_images, nine_images, captions, idx


class CocoImage(Dataset):
    def __init__(self, ann_dir, img_root, transform=None):
        self.transform = transform
        self.data = self.parse(Path(ann_dir), Path(img_root))

    @staticmethod
    def parse(ann_dir, img_root):
        ids = (
            np.load(ann_dir / "coco_train_ids.npy"),
            np.concatenate([
                np.load(ann_dir / "coco_restval_ids.npy"),
                np.load(ann_dir / "coco_dev_ids.npy"),
                np.load(ann_dir / "coco_test_ids.npy")
            ]),
        )
        coco = (
            pyCOCO(ann_dir / "captions_train2014.json"),
            pyCOCO(ann_dir / "captions_val2014.json"),
        )
        img_root = (img_root / "train2014", img_root / "val2014")

        data = {}
        for i in range(len(ids)):
            for idx in ids[i]:
                img_id = coco[i].anns[idx]["image_id"]
                img_file = img_root[i] / coco[i].loadImgs(img_id)[0]["file_name"]
                caption = coco[i].anns[idx]["caption"].strip()

                if img_id in data:  # one image has multiple captions
                    data[img_id]["captions"].append(caption)
                else:
                    data[img_id] = {
                        "image_id": img_id,
                        "image_file": img_file,
                        "captions": [caption, ]
                    }

        data = list(data.values())
        data.sort(key=lambda x: x["image_id"])

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index]["image_file"])
        image = image.convert("RGB")

        orig_image = self.transform(image)

        idx = self.data[index]["image_id"]
        file_name = self.data[index]["image_file"].stem
        # orig_image: (3, 224, 224)
        # idx: int (image id)
        return orig_image, file_name


class gqaImage(ImageFolder):
    def __init__(self, root, transform):
        self.transform = transform
        self.image_files = list(Path(root).glob("*.jpg"))
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index])
        image = image.convert("RGB")
        image = self.transform(image)
        file_name = self.image_files[index].stem
        return image, file_name


class CocoImage_for_mdetr(Dataset):
    def __init__(self, root, ann_dir, transform):
        root = Path(root)
        # ---------kkuhn-block------------------------------ # preprocess. Once done, just read the txt file below
        final_mixed_train_json = Path("ctx/datasets/coco_captions/annotations/OpenSource/final_mixed_train.json")
        with open(final_mixed_train_json, "r") as f:
            json_content = json.load(f)
        len(json_content["images"])
        self.data = []
        for image_info in json_content["images"]:
            self.data.append(image_info["file_name"])
        # find unique image ids
        self.data = list(set(self.data))

        self.data_only_coco = [i for i in self.data if "COCO" in i]
        # save to file
        with open("ctx/datasets/coco_captions/annotations/OpenSource/final_mixed_train_only_coco_unique.txt", "w") as f:
            for i in self.data_only_coco:
                f.write(i + "\n")
        # ---------kkuhn-block------------------------------

        # ---------kkuhn-block------------------------------ if txt file did not exist, create it by the code above
        root = Path("ctx/datasets/coco_captions/train2014")
        with open("ctx/datasets/coco_captions/annotations/OpenSource/final_mixed_train_only_coco_unique.txt", "r") as f:
            self.data_only_coco = f.read().splitlines()
        self.data_only_coco_Path = [root / i for i in self.data_only_coco]
        self.transform = transform
        # ---------kkuhn-block------------------------------

    def __len__(self):
        return len(self.data_only_coco_Path)

    def __getitem__(self, index):
        image = Image.open(self.data_only_coco_Path[index])
        image = image.convert("RGB")
        image = self.transform(image)
        file_name = self.data_only_coco_Path[index].stem
        return image, file_name


class gqaImage_for_mdetr(Dataset):
    def __init__(self, root, transform):
        # ---------kkuhn-block------------------------------ # preprocess. Once done, just read the txt file below
        final_mixed_train_json = Path("ctx/datasets/coco_captions/annotations/OpenSource/final_mixed_train.json")
        with open(final_mixed_train_json, "r") as f:
            json_content = json.load(f)
        len(json_content["images"])
        self.data = []
        for image_info in json_content["images"]:
            self.data.append(image_info["file_name"])
        # find unique image ids
        self.data = list(set(self.data))  # 88880

        self.data_only_gqa = [i for i in self.data if "COCO" not in i]  # 46380
        # save to file
        with open("ctx/datasets/coco_captions/annotations/OpenSource/final_mixed_train_only_gqa_unique.txt", "w") as f:
            for i in self.data_only_gqa:
                f.write(i + "\n")
        # ---------kkuhn-block------------------------------

        # ---------kkuhn-block------------------------------ # if txt file did not exist, create it by the code above
        root = Path("/home/szh2/datasets/gqa/images")
        with open("ctx/datasets/coco_captions/annotations/OpenSource/final_mixed_train_only_gqa_unique.txt", "r") as f:
            self.data_only_gqa = f.read().splitlines()
        self.data_only_gqa_Path = [root / i for i in self.data_only_gqa]
        self.transform = transform
        # ---------kkuhn-block------------------------------

    def __len__(self):
        return len(self.data_only_gqa_Path)

    def __getitem__(self, index):
        image = Image.open(self.data_only_gqa_Path[index])
        image = image.convert("RGB")
        image = self.transform(image)
        file_name = self.data_only_gqa_Path[index].stem
        return image, file_name


class flickrImage_for_mdetr(Dataset):
    def __init__(self, root, ann_dir, transform):
        # ---------kkuhn-block------------------------------ # preprocess. Once done, just read the txt file below
        ann_dir = Path(ann_dir)
        final_mixed_train_json = ann_dir / "final_flickr_separateGT_train.json"
        with open(final_mixed_train_json, "r") as f:
            json_content = json.load(f)
        len(json_content["images"])
        self.data = []
        for image_info in json_content["images"]:
            self.data.append(image_info["file_name"])
        # find unique image ids
        self.data = list(set(self.data))  # 29783

        self.data_only_flickr = self.data
        # save to file
        with open(ann_dir / "final_mixed_train_only_flickr_unique.txt", "w") as f:
            for i in self.data_only_flickr:
                f.write(i + "\n")
        # ---------kkuhn-block------------------------------

        # ---------kkuhn-block------------------------------ # if txt file did not exist, create it by the code above
        root = Path(root)
        with open(ann_dir / "final_mixed_train_only_flickr_unique.txt", "r") as f:
            self.data_only_flickr = f.read().splitlines()
        self.data_only_flickr_Path = [root / i for i in self.data_only_flickr]
        self.transform = transform
        # ---------kkuhn-block------------------------------

    def __len__(self):
        return len(self.data_only_flickr_Path)

    def __getitem__(self, index):
        image = Image.open(self.data_only_flickr_Path[index])
        image = image.convert("RGB")
        image = self.transform(image)
        file_name = self.data_only_flickr_Path[index].stem
        return image, file_name
