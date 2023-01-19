import json
import shutil

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from tqdm import tqdm


def expand2square_paint(pil_img, background_color, captions, filename):
    width, height = pil_img.size
    new_width = width + 200
    result = Image.new(pil_img.mode, (new_width, height), background_color)
    result.paste(pil_img)
    draw = ImageDraw.Draw(result)
    try:
        draw.text((width + 10, 10), captions, fill="black")
    except Exception as e:
        f_e.write(filename + "\t" + str(e) + "\n")
    return result


# ---------kkuhn-block------------------------------ # gqa
DATASET = "gqa"  # todo: must be set
image_folder = Path("/home/szh2/datasets/gqa/images")  # todo: must be set
images_captions_pair_json_file = Path('outputs/retrieved_captions_{}_100/image_caption_pairs.json'.format(DATASET))
painted_captions_dir = Path('ctx/outputs/painted_captions_{}'.format(DATASET))
if painted_captions_dir.exists():
    shutil.rmtree(painted_captions_dir)
painted_captions_dir.mkdir(parents=True, exist_ok=True)
error_log_file = Path('ctx/outputs/painted_captions_{}/error_log.txt'.format(DATASET))
f_e = open(error_log_file, 'w')
with open(images_captions_pair_json_file, 'r') as f:
    image_caption_pairs = json.loads(f.read())
iii = 0
for filename in tqdm(image_caption_pairs):
    captions = image_caption_pairs[filename]
    captions = "\n".join(captions)
    image_path = image_folder / (filename + ".jpg")
    image = Image.open(image_path)
    image = expand2square_paint(image, (255, 255, 255), captions, filename)
    image.save(painted_captions_dir / (filename + ".jpg"))
    iii += 1
    if iii > 100:
        break

f_e.close()
# ---------kkuhn-block------------------------------


# ---------kkuhn-block------------------------------ # coco
DATASET = "coco"  # todo: must be set
image_folder = Path("ctx/datasets/coco_captions")  # todo: must be set
images_captions_pair_json_file = Path('outputs/retrieved_captions_{}_100/image_caption_pairs.json'.format(DATASET))
painted_captions_dir = Path('ctx/outputs/painted_captions_{}'.format(DATASET))
if painted_captions_dir.exists():
    shutil.rmtree(painted_captions_dir)
painted_captions_dir.mkdir(parents=True, exist_ok=True)
error_log_file = Path('ctx/outputs/painted_captions_{}/error_log.txt'.format(DATASET))
f_e = open(error_log_file, 'w')
with open(images_captions_pair_json_file, 'r') as f:
    image_caption_pairs = json.loads(f.read())
iii = 0
for filename in tqdm(image_caption_pairs):
    captions = image_caption_pairs[filename]
    captions = "\n".join(captions)
    if "train" in filename:
        image_path = image_folder / "train2014" / (filename + ".jpg")
    elif "val" in filename:
        image_path = image_folder / "val2014" / (filename + ".jpg")
    else:
        raise Exception("unknown image path")
    image = Image.open(image_path)
    image = expand2square_paint(image, (255, 255, 255), captions, filename)
    image.save(painted_captions_dir / (filename + ".jpg"))
    iii += 1
f_e.close()
# ---------kkuhn-block------------------------------
