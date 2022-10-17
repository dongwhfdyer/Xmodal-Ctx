# save file name and file size
from pathlib import Path

imgFolder = Path(r"datasets/coco_captions/train2014")
with open("fileSize.txt", "a") as f:
    for imgPath in imgFolder.glob("*.jpg"):

        f.write(imgPath.name + " " + str(imgPath.stat().st_size) + "\n")
        # write file hash
