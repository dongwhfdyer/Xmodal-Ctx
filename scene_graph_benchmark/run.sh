dataDir="datasets"
modelDir="downloads/vinvl_vg_x152c4.pth"
configFile="sgg_configs/vgattr/vinvl_x152c4.yaml"
batchSize=4
python tools/test_sg_net.py --config-file $configFile TEST.IMS_PER_BATCH $batchSize MODEL.WEIGHT $modelDir MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 DATA_DIR $dataDir TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True TEST.OUTPUT_FEATURE True
#python tools/test_sg_net.py --config-file sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 2 MODEL.WEIGHT downloads/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 DATA_DIR datasets TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True TEST.OUTPUT_FEATURE True
