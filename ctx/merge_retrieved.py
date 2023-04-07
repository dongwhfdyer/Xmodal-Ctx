import h5py
import numpy as np
from tqdm import tqdm

# # ---------kkuhn-block------------------------------ # coco2014
# outputpath = "outputs/coco_crop/coco_all.hdf5"
#
# file1 = "outputs/coco_crop/txt_ctx_0.hdf5"
# file2 = "outputs/coco_crop/txt_ctx_1.hdf5"
# file3 = "outputs/coco_crop_val/txt_ctx_0.hdf5"
# file4 = "outputs/coco_crop_val/txt_ctx_1.hdf5"
# file5 = "outputs/coco_crop_test/txt_ctx_0.hdf5"
# file6 = "outputs/coco_crop_test/txt_ctx_1.hdf5"
#
# # read hdf5 and save to the outputpath
# with h5py.File(outputpath, "w") as f:
#     for file in [file1, file2, file3, file4, file5, file6]:
#         with h5py.File(file, "r") as f1:
#             for gg in tqdm(f1.keys()):
#                 g_name = gg[:-4]
#                 for gg2 in f1[gg].keys():
#                     g = f.create_group(g_name + "/" + gg2)
#                     g.create_dataset("features", data=f1[gg][gg2]["features"])
#                     g.create_dataset("texts", data=f1[gg][gg2]["texts"])
#                     pass
# # ---------kkuhn-block------------------------------

# ---------kkuhn-block------------------------------ # coco2017
outputpath = "outputs/coco2017all_crop/coco2017_crop_caps.hdf5"

file1 = "outputs/coco2017all_crop/txt_ctx_0.hdf5"
file2 = "outputs/coco2017all_crop/txt_ctx_1.hdf5"

# read hdf5 and save to the outputpath
with h5py.File(outputpath, "w") as f:
    for file in [file1, file2]:
        with h5py.File(file, "r") as f1:
            for gg in tqdm(f1.keys()):
                g_name = gg
                for gg2 in f1[gg].keys():
                    g = f.create_group(g_name + "/" + gg2)
                    g.create_dataset("features", data=f1[gg][gg2]["features"])
                    g.create_dataset("texts", data=f1[gg][gg2]["texts"])
# ---------kkuhn-block------------------------------
