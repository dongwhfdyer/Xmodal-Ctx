import h5py

img_id = 531912
top_k = 4
crop_n = 2
with h5py.File("outputs/retrieved_captions/txt_ctx.hdf5", "r") as f:
    features = f[f"{img_id}/five/features"][crop_n, :top_k]
    scores = f[f"{img_id}/five/scores"][crop_n, :top_k]
    texts = f[f"{img_id}/five/texts"][crop_n, :top_k]

print(features.shape)  # (4, 512)
print(scores.shape)  # (4,)
print(len(texts))  # 4
print(scores)
print(features)
print(texts)