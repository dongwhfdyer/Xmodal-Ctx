# kuhn notes
[Beam search article](https://blog.csdn.net/xyz1584172808/article/details/89220906?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166509858716782391879316%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166509858716782391879316&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-3-89220906-null-null.142^v51^control,201^v3^control_1&utm_term=BeamSearch)
# Train M<sup>2</sup> image captioning model

## Setup

```bash
mkdir datasets && cd datasets

# Download COCO caption annotations
gdown --fuzzy https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing
unzip annotations.zip
rm annotations.zip

# Download object features
wget https://www.dropbox.com/s/0h67c6ezwnderbd/oscar.hdf5
wget https://www.dropbox.com/s/hjh7shr5zvaz3gj/vinvl.hdf5

# Link cross-modal context
ln -s ../../ctx/outputs/image_features/vis_ctx.hdf5
ln -s ../../ctx/outputs/retrieved_captions/txt_ctx.hdf5

```

## Training

The training is conducted on 1 A40 GPUs and takes approximately 4 days.

- Train M<sup>2</sup> + weaker Visual Genome object features + our cross-modal context on GPU #0 (or any available GPU on your machine).

    ```Bash
    python train.py --obj_file oscar.hdf5 --devices 0
    ```

- Train M<sup>2</sup> + stronger VinVL object features + our cross-modal context on GPU #1 (or any available GPU on your machine).

    ```Bash
    python train.py --obj_file vinvl.hdf5 --devices 1
    ```

## Results

Using the weaker Visual Genome object features (`oscar.hdf5`).

| Method | XModal Ctx | B-1 | B-4 | M | R | C | S |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M^2 (paper) | N |  80.8 | 39.1 | 29.1 | 58.4 | 131.2 | 22.6 |
| M^2 (codebase) | N | 80.2 | 38.4 | 29.1 | 58.4 | 128.7 | 22.9 |
| __Ours__ | __Y__ | __81.5__ | __39.7__ | __30.0__ | __59.5__ | __135.9__ | __23.7__ |

Using the stronger VinVL object features (`vinvl.hdf5`).

| Method | XModal Ctx | Object<br/>Features | B-1 | B-4 | M | R | C | S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M^2 | N | VG | 80.2 | 38.4 | 29.1 | 58.4 | 128.7 | 22.9 |
| M^2 | N | VinVL | 82.7 | 40.5 | 29.9 | 59.9 | 135.9 | 23.5 |
| Ours | Y | VG | 81.5 | 39.7 | 30.0 | 59.5 | 135.9 | 23.7 |
| __Ours__ | __Y__ | __VinVL__ | __83.4__ | __41.4__ | __30.4__ | __60.4__ | __139.9__ | __24.0__ |

## Citations

Please cite our work if you find this repo useful.

```BibTeX
@inproceedings{kuo2022pretrained,
    title={Beyond a Pre-Trained Object Detector: Cross-Modal Textual and Visual Context for Image Captioning},
    author={Chia-Wen Kuo and Zsolt Kira},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2022}
}
```

This codebase is built upon the official implementation of [M<sup>2</sup>](https://github.com/aimagelab/meshed-memory-transformer). Consider citing thier work if you find this repo useful.

```BibTeX
@inproceedings{cornia2020m2,
    title={{Meshed-Memory Transformer for Image Captioning}},
    author={Cornia, Marcella and Stefanini, Matteo and Baraldi, Lorenzo and Cucchiara, Rita},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2020}
}
```
