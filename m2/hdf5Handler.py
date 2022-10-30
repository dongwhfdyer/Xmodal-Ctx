import h5py

# #---------kkuhn-block------------------------------ # try to read the hdf5 file
# # obj_file = r"datasets/oscar.hdf5"
# obj_file = r"../scene_graph_benchmark/featureOutputs/puzzleCOCOFeature.hdf5"
# obj = h5py.File(obj_file, "r")
# print(obj.keys())
# dd = obj["99999"]
# print(dd.keys())
# ddd=dd['obj_features']
# dddds=obj["99999"]['num_boxes'].value
# dddd = obj["99999/obj_features"][:]
#
# #---------kkuhn-block------------------------------

#---------kkuhn-block------------------------------ # transfer the hdf5 file to a file
obj_file1 = r"../scene_graph_benchmark/featureOutputs/puzzleCOCOFeature.hdf5"
obj_file2 = r"../scene_graph_benchmark/featureOutputs/puzzleCOCOFeatureVal.hdf5"
obj_file3 = r"../scene_graph_benchmark/featureOutputs/allPuzzleCOCOFeature.hdf5"
obj1 = h5py.File(obj_file1, "r")
obj2 = h5py.File(obj_file2, "r")
obj3 = h5py.File(obj_file3, "w")
# iterate over the keys of the second file and copy them to the first file
for key in obj1.keys():
    obj3.create_group(key)
    obj3[key]['num_boxes'] = obj1[key]['num_boxes'].value
    obj3[key]['obj_features'] = obj1[key]['obj_features'][:]
obj1.close()
for key in obj2.keys():
    obj3.create_group(key)
    obj3[key]['num_boxes'] = obj2[key]['num_boxes'].value
    obj3[key]['obj_features'] = obj2[key]['obj_features'][:]
obj2.close()
obj3.close()
#---------kkuhn-block------------------------------


