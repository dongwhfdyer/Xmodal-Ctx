import h5py
import numpy as np


def readHDF5File(hdf5File):
    with h5py.File(hdf5File, 'r') as f:
        return f


def readOneFromHDF5(objPath, ind):
    obj = h5py.File(objPath, 'r')
    print(obj.keys())
    len(obj.keys())
    objId = list(obj.keys())
    oneExp = obj[objId[ind]]
    oneExpValues = list(oneExp.values())
    numBoxes = np.array(oneExpValues[0])
    objFeatures = np.array(oneExpValues[1])
    obj.close()
    return numBoxes, objFeatures


def writeToHDF5(objPath, numBoxes, objFeatures):
    obj = h5py.File(objPath, 'w')
    obj.create_group('0')
    obj['0']['num_boxes'] = numBoxes
    obj['0']['obj_features'] = objFeatures
    obj.close()


def testReadData():
    # objFilePath = r"/home/pcl/kuhn/Xmodal-Ctx/scene_graph_benchmark/temp/rubb.hdf5"
    # objFilePath = r"temp/rubb.hdf5"
    oscarObjFilePath = r"datasets/oscar.hdf5"
    vinvlObjFilePath = r"datasets/vinvl.hdf5"
    puzzlecocoFilePath = r"datasets/puzzleCOCOFeature.hdf5"
    # oscarObj = h5py.File(oscarObjFilePath, 'r')
    puzzleObj = h5py.File(puzzlecocoFilePath, 'r')
    vinvlObj = h5py.File(vinvlObjFilePath, 'r')  # 123287 images
    # ss = numpy.array(vinvlObj['42/obj_features'])
    # sss = numpy.array(oscarObj['42/obj_features'])

    # print(oscarObj.keys())
    # print(vinvlObj.keys())
    # print(puzzleObj.keys())
    with open("temp/puzzleObj.txt", 'w') as f:
        f.write(str(puzzleObj.keys()))
    with open("temp/vinvlObj.txt", 'w') as f:
        f.write(str(vinvlObj.keys()))

    puzzleObj.close()
    # oscarObj.close()
    vinvlObj.close()
    # numBoxes, objFeatures = readOneFromHDF5(objFilePath, 0)
    # obj_filePath3 = r"temp/rubb.hdf5"
    # writeToHDF5(obj_filePath3, numBoxes, objFeatures)
    # numBoxes1, objFeatures1 = readOneFromHDF5(obj_filePath3, 0)

    print("--------------------------------------------------")


if __name__ == '__main__':
    testReadData()
    # # obj_file = r"temp/rubb.hdf5"
    # # obj = h5py.File(obj_file, "w")
    # dataset_root = Path(p_opt.dataset_root)
    # # Create the dataset
    # object_field = ImageDetectionsField(
    #     obj_file=Path(dataset_root) / p_opt.obj_file,
    #     max_detections=50, preload=p_opt.preload
    # )
    # text_field = TextField(
    #     init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
    #     remove_punctuation=True, nopoints=False
    # )
    # txt_ctx_filed = TxtCtxField(
    #     ctx_file=dataset_root / "txt_ctx.hdf5", k=p_opt.topk, preload=p_opt.preload
    # )
    # vis_ctx_filed = VisCtxField(
    #     ctx_file=dataset_root / "vis_ctx.hdf5", preload=p_opt.preload
    # )
    # fields = {
    #     "object": object_field, "text": text_field, "img_id": RawField(),
    #     "txt_ctx": txt_ctx_filed, "vis_ctx": vis_ctx_filed
    # }
    # dset = dataset_root / "annotations"
    # dataset = COCO(fields, dset, dset)
    # # each dataset has many examples
    # # example = {
    # #     "img_id": img_id,
    # #     "object": img_id,
    # #     "text": caption,
    # #     "txt_ctx": img_id,
    # #     "vis_ctx": img_id
    # # }
    # train_dataset, val_dataset, test_dataset = dataset.splits
    # ss = dataset.__getitem__(0)["img_id"]

    # obj_file = r"datasets/vinvl.hdf5"
    # obj = h5py.File(obj_file, "r")
    # objTest = h5py.File(obj_file, "w")

    # boxes, features = readOneFromHDF5(obj, 8)
    # writeOneToHDF5("0", 1, 2)
