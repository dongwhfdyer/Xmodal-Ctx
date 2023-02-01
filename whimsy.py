import copy
import itertools
import json
from pathlib import Path

from PIL import Image
from torchvision.transforms import functional as F

import torch


def testExpandTensor():
    tt = torch.zeros([25, 60, 512])
    # posEnd = torch.Tensor([0.3, 0.5, 0.6, 0.2])
    # posEnd =torch.Tensor([
    #             [0.2, 0.2, 0.4, 0.4, ],
    #             [0.2, 0.2, 0.4, 0.4, ],
    #             [0.2, 0.2, 0.4, 0.4, ],
    #             [0.5, 0.5, 0.4, 0.4, ],
    #             [0.5, 0.5, 0.4, 0.4, ],
    #             [0.5, 0.5, 0.4, 0.4, ],
    #             [0.8, 0.8, 0.4, 0.4, ],
    #             [0.8, 0.8, 0.4, 0.4, ],
    #             [0.8, 0.8, 0.4, 0.4, ], ]
    #         )
    posEnd = torch.Tensor([[0.3000, 0.3000, 0.6000, 0.6000],
                           [0.7000, 0.3000, 0.6000, 0.6000],
                           [0.3000, 0.7000, 0.6000, 0.6000],
                           [0.7000, 0.7000, 0.6000, 0.6000],
                           [0.5000, 0.5000, 0.6000, 0.6000]])

    temp1 = posEnd.unsqueeze(0)
    temp2 = temp1.repeat(25, 1, 12)
    temp3 = temp2.reshape(25, 60, 4)
    wholeProc = posEnd.unsqueeze(0).repeat(25, 1, 12).reshape(25, 60, 4)

    temp3 = torch.concat([tt, temp3], -1)
    temp4 = temp3[0]
    temp5 = temp4[:12]
    temp6 = temp4[12:24]
    temp7 = temp4[24:36]
    temp8 = temp4[36:48]
    print("--------------------------------------------------")


def nine_crop(ratio=0.4):
    # def nine_crop(image, ratio=0.4):

    # w, h = 224, 224
    w, h = 512, 512
    # w, h = 768, 512
    origW = copy.copy(w)
    origH = copy.copy(h)

    t = (0, int((0.5 - ratio / 2) * h), int((1.0 - ratio) * h))
    b = (int(ratio * h), int((0.5 + ratio / 2) * h), h)
    l = (0, int((0.5 - ratio / 2) * w), int((1.0 - ratio) * w))
    r = (int(ratio * w), int((0.5 + ratio / 2) * w), w)
    h, w = list(zip(t, b)), list(zip(l, r))

    images = []
    hwInfo = []
    centerInfo = []
    for s in itertools.product(h, w):
        h, w = s
        top, left = h[0], w[0]
        height, width = h[1] - h[0], w[1] - w[0]
        centerX = top + width / 2
        centerY = top + height / 2
        centerXP = centerX / origW
        centerYP = centerY / origH
        heightP = height / origH
        widthP = width / origW
        # keep two decimal places
        centerXP, centerYP, heightP, widthP = round(centerXP, 2), round(centerYP, 2), round(heightP, 2), round(widthP, 2)
        hwInfo.append((top, left, height, width))
        centerInfo.append((centerXP, centerYP, heightP, widthP))

        # images.append(F.crop(image, top, left, height, width))
    # print centerInfo
    for l in centerInfo:
        print('[', end='')
        for i in l:
            print(i, end=",")
        print('],')


def rubb1():
    dd = torch.Tensor(([67.2, 67.2, 134.4, 134.4],
                       [156.8, 67.2, 134.4, 134.4],
                       [67.2, 156.8, 134.4, 134.4],
                       [156.8, 156.8, 134.4, 134.4],
                       [112, 112, 134.4, 134.4],))
    print(dd)
    ddd = dd / 224
    print(ddd)


def testhdf5saving():
    import h5py
    objPath = r"temp/rubb.hdf5"

    def saveByBatch(featureDict):
        obj = h5py.File(objPath, 'a')
        for img_id, feature in featureDict.items():
            img_id = str(img_id)
            obj.create_group(img_id)
            obj[img_id]['num_boxes'] = len(feature.get_field('box_features'))
            obj[img_id]['obj_features'] = feature.get_field('box_features').numpy()
        obj.close()

    obj = h5py.File(objPath, 'a')
    obj.create_group('1')
    obj['1']['num_boxes'] = 1
    obj['1']['obj_features'] = torch.Tensor([1, 2, 3, 4]).numpy()
    obj.close()

    obj = h5py.File(objPath, 'a')
    obj.create_group('2')
    obj['2']['num_boxes'] = 2
    obj['2']['obj_features'] = torch.Tensor([1, 2, 3, 4, 5, 6]).numpy()
    obj.close()

    obj = h5py.File(objPath, 'r')
    print(obj.keys())
    obj.close()


def test_onehots():
    import torch as t
    import numpy as np

    batch_size = 8
    class_num = 10199
    label = np.random.randint(0, class_num, size=(batch_size, 1))
    label = t.LongTensor(label)

    y_one_hot = t.zeros(batch_size, class_num).scatter_(1, label, 1)
    print(y_one_hot)


def test_onehot_v2():
    import torch
    import torch.nn.functional as Fun
    batchSize = 50
    wordNum = 22
    classNum = 10199
    A = torch.randint(0, classNum, (batchSize, wordNum, classNum))
    output = Fun.one_hot(A.argmax(dim=2), classNum)
    sample1 = output[0][0].argmax(dim=0)
    print(output)


def test_onehot_v3():
    import torch
    import torch.nn.functional as Fun
    batchSize = 50
    wordNum = 22
    classNum = 10199
    A = torch.randint(0, classNum, (batchSize, wordNum))
    output = Fun.one_hot(A, classNum)
    print(output)


from nltk.corpus import wordnet as wn


def test_wordnet():
    # download wordnet
    import nltk
    # nltk.download('wordnet')
    word = wn.synsets("word")  # 同义词集
    # 返回的 Synset('word.n.01')表示单词word在名词中的第1个释义;
    print("同义词集：", word)
    print("释义：", word[0].definition())
    print("例句：", word[0].examples())
    print("同一涵义对应的多个词条：", word[0].lemmas()[0])
    print("同一涵义对应的多个词条的名称：", word[0].lemma_names())


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

    print("--------------------------------------------------")


if __name__ == '__main__':
    # test_data_integrity()
    add_two_json_together()
    pass
