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


if __name__ == '__main__':
    permutations9 = np.load("datasets/permutations_hamming_max_64.npy")
    imgTrainFolder = Path("datasets/coco_captions/train2014")
    imgValFolder = Path("datasets/coco_captions/val2014")
    imgTrainDestFolder = Path("datasets/coco_captions/train2014Random9Crop")
    imgValDestFolder = Path("datasets/coco_captions/val2014Random9Crop")
    imgErrorFolder = Path("datasets/coco_captions/error")
    imgResumedFolder = Path("datasets/coco_captions/resumed")
    annotationTrainFile = "datasets/coco_captions/trainRandom9Info.txt"
    annotationValFile = "datasets/coco_captions/valRandom9Info.txt"

    delete_folders(imgTrainDestFolder, imgValDestFolder, imgErrorFolder, imgResumedFolder)
    create_folders(imgTrainDestFolder, imgValDestFolder, imgErrorFolder, imgResumedFolder)
    tackleOneFolder(imgTrainFolder, imgTrainDestFolder, annotationTrainFile)
    # tackleOneFolderReverse(imgTrainDestFolder, imgResumedFolder)
    tackleOneFolder(imgValFolder, imgValDestFolder, annotationValFile)
