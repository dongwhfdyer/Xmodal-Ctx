# if no datasets folder, create
if [ ! -d "datasets" ]; then
  mkdir datasets
fi

#    permutations9 = np.load("datasets/permutations_hamming_max_64.npy")
#    imgTrainFolder = Path("datasets/coco_captions/train2014")
#    imgValFolder = Path("datasets/coco_captions/val2014")
#    imgTrainDestFolder = Path("datasets/coco_captions/train2014Random9Crop")
#    imgValDestFolder = Path("datasets/coco_captions/val2014Random9Crop")
#    imgErrorFolder = Path("datasets/coco_captions/error")
#    imgResumedFolder = Path("datasets/coco_captions/resumed")
#    annotationTrainFile = "datasets/coco_captions/annotations/trainRandom9Info.json"
#    annotationValFile = "datasets/coco_captions/annotations/valRandom9Info.json"
ln -s ~/datasets/coco/train2014 datasets/coco_captions/train2014
ln -s ~/datasets/coco/val2014 datasets/coco_captions/val2014
ln -s ~/datasets/coco/annotations datasets/coco_captions/annotations

