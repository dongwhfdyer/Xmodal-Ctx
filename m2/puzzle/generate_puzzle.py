import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def crop9(imgPath, permutation):
    # crop the image into 9 blocks, shuffle them and reassemble them
    try:
        img = Image.open(imgPath)
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        w = w // 3
        h = h // 3
        img = np.array(img, np.float32)
        One = img[0:h, 0:w, :]
        Two = img[0:h, w:2 * w, :]
        Three = img[0:h, 2 * w:3 * w, :]
        Four = img[h:2 * h, 0:w, :]
        Five = img[h:2 * h, w:2 * w, :]
        Six = img[h:2 * h, 2 * w:3 * w, :]
        Seven = img[2 * h:3 * h, 0:w, :]
        Eight = img[2 * h:3 * h, w:2 * w, :]
        Nine = img[2 * h:3 * h, 2 * w:3 * w, :]

        blocks = [One, Two, Three, Four, Five, Six, Seven, Eight, Nine]
        newBlocks = [blocks[i] for i in permutation]
        img = np.concatenate((np.concatenate((newBlocks[0], newBlocks[1], newBlocks[2]), axis=1),
                              np.concatenate((newBlocks[3], newBlocks[4], newBlocks[5]), axis=1),
                              np.concatenate((newBlocks[6], newBlocks[7], newBlocks[8]), axis=1)), axis=0)
        img = Image.fromarray(np.uint8(img))
    except Exception as e:
        print(e)
        print(imgPath)
        # show the original image
        img = Image.open(imgPath)
        img.save(imgErrorFolder / imgPath.name)
        return None
    return img


def resume9blocks(imgPath, permutation):
    # crop the image into 9 blocks, shuffle them and reassemble them
    img = Image.open(imgPath)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    w = w // 3
    h = h // 3
    img = np.array(img, np.float32)
    img = np.array(img, np.float32)
    One = img[0:h, 0:w, :]
    Two = img[0:h, w:2 * w, :]
    Three = img[0:h, 2 * w:3 * w, :]
    Four = img[h:2 * h, 0:w, :]
    Five = img[h:2 * h, w:2 * w, :]
    Six = img[h:2 * h, 2 * w:3 * w, :]
    Seven = img[2 * h:3 * h, 0:w, :]
    Eight = img[2 * h:3 * h, w:2 * w, :]
    Nine = img[2 * h:3 * h, 2 * w:3 * w, :]

    blocks = [One, Two, Three, Four, Five, Six, Seven, Eight, Nine]
    permutation = list(permutation)
    # deep copy blocks
    newBlocks = [blocks[i].copy() for i in range(9)]
    for i in range(9):
        newBlocks[i] = blocks[permutation.index(i)]
    img = np.concatenate((np.concatenate((newBlocks[0], newBlocks[1], newBlocks[2]), axis=1),
                          np.concatenate((newBlocks[3], newBlocks[4], newBlocks[5]), axis=1),
                          np.concatenate((newBlocks[6], newBlocks[7], newBlocks[8]), axis=1)), axis=0)
    img = Image.fromarray(np.uint8(img))
    return img


def resume9blocks_v2(imgPath, permutation):
    """
    This one allow replicate permutation.
    """
    # crop the image into 9 blocks, shuffle them and reassemble them
    img = Image.open(imgPath)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    w = w // 3
    h = h // 3
    img = np.array(img, np.float32)
    img = np.array(img, np.float32)
    One = img[0:h, 0:w, :]
    Two = img[0:h, w:2 * w, :]
    Three = img[0:h, 2 * w:3 * w, :]
    Four = img[h:2 * h, 0:w, :]
    Five = img[h:2 * h, w:2 * w, :]
    Six = img[h:2 * h, 2 * w:3 * w, :]
    Seven = img[2 * h:3 * h, 0:w, :]
    Eight = img[2 * h:3 * h, w:2 * w, :]
    Nine = img[2 * h:3 * h, 2 * w:3 * w, :]

    blocks = [One, Two, Three, Four, Five, Six, Seven, Eight, Nine]
    permutation = list(permutation)
    # deep copy blocks
    newBlocks = [blocks[i].copy() for i in range(9)]
    for p, i in enumerate(permutation):
        newBlocks[i] = blocks[p]
    img = np.concatenate((np.concatenate((newBlocks[0], newBlocks[1], newBlocks[2]), axis=1),
                          np.concatenate((newBlocks[3], newBlocks[4], newBlocks[5]), axis=1),
                          np.concatenate((newBlocks[6], newBlocks[7], newBlocks[8]), axis=1)), axis=0)
    img = Image.fromarray(np.uint8(img))
    return img


def get_one_image(img_list):
    max_width = 0
    total_height = 200  # padding
    for img in img_list:
        if img.shape[1] > max_width:
            max_width = img.shape[1]
        total_height += img.shape[0]

    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in img_list:
        # add an image to the final array and increment the y coordinate
        image = np.hstack((image, np.zeros((image.shape[0], max_width - image.shape[1], 3))))
        final_image[current_y:current_y + image.shape[0], :, :] = image
        current_y += image.shape[0]
    return final_image


def tackleOneFolder(imgFolder, destFolder, annotationfile):
    tbar = tqdm(list(imgFolder.glob("*.jpg")))
    f = open(annotationfile, 'w')
    content = ""
    iter = 0
    for imgPath in imgFolder.glob("*.jpg"):
        iter += 1
        tbar.update(1)
        randomId = np.random.randint(0, 64)
        croppedImg = crop9(imgPath, permutations9[randomId])
        croppedImg.save(destFolder / imgPath.name)
        content += imgPath.stem + " " + str(randomId) + "\n"
        if iter % 1000 == 0:
            f.write(content)
            content = ""

    f.write(content)
    f.close()


def tackleOneFolder_readVersion(cocoPath, annotationfile):
    cocoPath = Path(cocoPath)
    with open(annotationfile, 'r') as f:
        content = f.readlines()
    tbar = tqdm(content)
    for line in content:
        tbar.update(1)
        imgName, permutation_order = line.split()
        cls = "train2014" if "train" in imgName else "val2014"
        imgPath = cocoPath / cls / (imgName + ".jpg")
        croppedImg = crop9(imgPath, permutations9[int(permutation_order)])
        cls_target = "train2014Random9Crop" if "train" in imgName else "val2014Random9Crop"
        savePath = cocoPath / cls_target / (imgName + ".jpg")
        croppedImg.save(savePath)


def tackleOneFolderReverse(imgFolder, destFolder, annotationfile):
    f = open(annotationfile, 'r')
    for line in f.readlines():
        imgStem, permutationID = f.readline().strip().split(' ')
        permutation = permutations9[int(permutationID)]
        imgPath = imgFolder / (imgStem + ".jpg")
        resumedImg = resume9blocks(imgPath, permutation)
        resumedImg.save(destFolder / imgPath.name)
    f.close()


def delete_folders(*folder_path):
    for folder in folder_path:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def create_folders(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def generatePuzzleCOCODataset():
    """
    Generate a COCO dataset with 9 blocks puzzle.
    It's generating without any annotation. It will output an annotation file, and puzzle dataset.
    """
    permutations9 = np.load("datasets/permutations_hamming_max_64.npy")
    imgTrainFolder = Path("datasets/coco/train2014")
    imgValFolder = Path("datasets/coco/val2014")
    imgTrainDestFolder = Path("datasets/coco/train2014Random9Crop")
    imgValDestFolder = Path("datasets/coco/val2014Random9Crop")
    imgErrorFolder = Path("datasets/coco/error")
    imgResumedFolder = Path("datasets/coco/resumed")
    annotationTrainFile = "datasets/coco/trainRandom9Info.txt"
    annotationValFile = "datasets/coco/valRandom9Info.txt"

    delete_folders(imgTrainDestFolder, imgValDestFolder, imgErrorFolder, imgResumedFolder)
    create_folders(imgTrainDestFolder, imgValDestFolder, imgErrorFolder, imgResumedFolder)
    tackleOneFolder(imgTrainFolder, imgTrainDestFolder, annotationTrainFile)
    # tackleOneFolderReverse(imgTrainDestFolder, imgResumedFolder)
    tackleOneFolder(imgValFolder, imgValDestFolder, annotationValFile)


def generatePuzzleCOCODataset_v2():
    """
    Generate a COCO dataset with 9 blocks puzzle.
    It's generating with an annotation file. It will output the puzzle dataset according to the annotation file.
    """
    global permutations9
    permutations9 = np.load("datasets/permutations_hamming_max_64.npy")
    cocoImgFolder = Path("datasets/coco")
    imgTrainDestFolder = Path("datasets/coco/train2014Random9Crop")
    imgValDestFolder = Path("datasets/coco/val2014Random9Crop")
    imgErrorFolder = Path("datasets/coco/error")
    imgResumedFolder = Path("datasets/coco/resumed")
    annotationTrainvalFile = "datasets/annotations/trainvalRandom9Info.txt"

    delete_folders(imgTrainDestFolder, imgValDestFolder, imgErrorFolder, imgResumedFolder)
    create_folders(imgTrainDestFolder, imgValDestFolder, imgErrorFolder, imgResumedFolder)
    tackleOneFolder_readVersion(cocoImgFolder, annotationTrainvalFile)


if __name__ == '__main__':
    generatePuzzleCOCODataset_v2()
