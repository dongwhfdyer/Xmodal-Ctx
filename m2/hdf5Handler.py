import h5py

# obj_file = r"datasets/oscar.hdf5"
obj_file = r"../scene_graph_benchmark/featureOutputs/puzzleCOCOFeature.hdf5"
obj = h5py.File(obj_file, "r")
print(obj.keys())
dd = obj["99999"]
print(dd.keys())
ddd=dd['obj_features']
dddd = obj["99999/obj_features"]

print("--------------------------------------------------")