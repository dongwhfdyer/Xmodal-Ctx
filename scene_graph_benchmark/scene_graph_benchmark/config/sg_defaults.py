# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Scene Graph Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# -----------------------------------------------------------------------------
# Attribute Head Options
# -----------------------------------------------------------------------------
_C.MODEL.ATTRIBUTE_ON = False
_C.MODEL.ROI_ATTRIBUTE_HEAD = CN()
_C.MODEL.ROI_ATTRIBUTE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
_C.MODEL.ROI_ATTRIBUTE_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_ATTRIBUTE_HEAD.PREDICTOR = "AttributeRCNNPredictor"
_C.MODEL.ROI_ATTRIBUTE_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_ATTRIBUTE_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_ATTRIBUTE_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES = 401
_C.MODEL.ROI_ATTRIBUTE_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_ATTRIBUTE_HEAD.CLS_EMD_DIM = 256
_C.MODEL.ROI_ATTRIBUTE_HEAD.ATTR_EMD_DIM = 512
_C.MODEL.ROI_ATTRIBUTE_HEAD.MAX_NUM_ATTR_PER_IMG = 100
_C.MODEL.ROI_ATTRIBUTE_HEAD.MAX_NUM_ATTR_PER_OBJ = 16
_C.MODEL.ROI_ATTRIBUTE_HEAD.POSTPROCESS_ATTRIBUTES_THRESHOLD = 0.0
_C.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT = 0.5
# Dilation
# _C.MODEL.ROI_ATTRIBUTE_HEAD.DILATION = 1
# GN
# _C.MODEL.ROI_ATTRIBUTE_HEAD.USE_GN = False

# -----------------------------------------------------------------------------
# Relation Head Options
# -----------------------------------------------------------------------------
_C.MODEL.RELATION_ON = False
_C.MODEL.USE_FREQ_PRIOR = False
_C.MODEL.FREQ_PRIOR = "visualgenome/label_danfeiX_clipped.freq_prior.npy"
_C.MODEL.ROI_RELATION_HEAD = CN()
_C.MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED = False
_C.MODEL.ROI_RELATION_HEAD.DETECTOR_BOX_THRESHOLD = 0.0
_C.MODEL.ROI_RELATION_HEAD.POSTPROCESS_METHOD = 'constrained'
_C.MODEL.ROI_RELATION_HEAD.POSTPROCESS_SCORE_THRESH = 0.00001
_C.MODEL.ROI_RELATION_HEAD.FORCE_RELATIONS = False
_C.MODEL.ROI_RELATION_HEAD.ALGORITHM = "sg_baseline"
_C.MODEL.ROI_RELATION_HEAD.MODE = 'sgdet'
_C.MODEL.ROI_RELATION_HEAD.USE_RELPN = False
_C.MODEL.ROI_RELATION_HEAD.POST_RELPN_PREPOSALS = 512
_C.MODEL.ROI_RELATION_HEAD.USE_BIAS = False
_C.MODEL.ROI_RELATION_HEAD.USE_ONLINE_OBJ_LABELS = False
_C.MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP = True
_C.MODEL.ROI_RELATION_HEAD.UPDATE_BOX_REG = False
_C.MODEL.ROI_RELATION_HEAD.TRIPLETS_PER_IMG = 100
_C.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIRelationFeatureExtractor"
_C.MODEL.ROI_RELATION_HEAD.PREDICTOR = "FastRCNNRelationPredictor"
_C.MODEL.ROI_RELATION_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_RELATION_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_RELATION_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_RELATION_HEAD.NUM_CLASSES = 51
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_RELATION_HEAD.MLP_HEAD_DIM = 1024
# GN
_C.MODEL.ROI_RELATION_HEAD.USE_GN = False
# Dilation
_C.MODEL.ROI_RELATION_HEAD.DILATION = 1
_C.MODEL.ROI_RELATION_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.ROI_RELATION_HEAD.NUM_STACKED_CONVS = 4
_C.MODEL.ROI_RELATION_HEAD.SHARE_CONV_BACKBONE = True
_C.MODEL.ROI_RELATION_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
_C.MODEL.ROI_RELATION_HEAD.SEPERATE_SO_FEATURE_EXTRACTOR = False

_C.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION = 0.25

# free object detection or not when training relation head
_C.MODEL.ROI_RELATION_HEAD.BACKBONE_FREEZE_PARAMETER = True
_C.MODEL.ROI_RELATION_HEAD.RPN_FREEZE_PARAMETER = True
_C.MODEL.ROI_RELATION_HEAD.ROI_BOX_HEAD_FREEZE_PARAMETER = True

# neural motif meta params
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF = CN()
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_CLASSES_FN = 'visualgenome/label_danfeiX_clipped.obj_classes.txt'
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.REL_CLASSES_FN = 'visualgenome/label_danfeiX_clipped.rel_classes.txt'
#
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.USE_TANH = False

# neural motif model params
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.ORDER = 'confidence'
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.NUM_OBJS = 64
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.EMBED_DIM = 100
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.HIDDEN_DIM = 256
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_LSTM_NUM_LAYERS = 2
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.EDGE_LSTM_NUM_LAYERS = 4
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.DROPOUT = 0.0
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_FEAT_TO_DECODER = False
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_FEAT_TO_EDGE = False
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.POS_BATCHNORM_MOMENTUM = 0.001
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.POS_EMBED_DIM = 128
# glove data path
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.GLOVE_PATH = 'glove/'
# only set to true in debug mode!
# it does: 1) make word vector all random since loading time is long.
_C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.DEBUG = False

# _C.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_FEAT_DIM = 1024
_C.MODEL.ROI_RELATION_HEAD.CONCATENATE_PROPOSAL_GT = False

# reldn contrastive loss meta params
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS = CN()
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_FLAG = False
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_SAMPLE_SIZE = 128
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_REL_SIZE_PER_IM = 512
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_REL_FRACTION = 0.25
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.FG_THRESH = 0.5
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_BG = True
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.BG_THRESH_HI = 0.5
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.BG_THRESH_LO = 0.0
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_SPATIAL_FEAT = False
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_FREQ_BIAS = True
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_LOSS = True
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_MARGIN = 0.2
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_WEIGHT = 1.0
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_SO_AWARE_LOSS = True
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_SO_AWARE_MARGIN = 0.2
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_SO_AWARE_WEIGHT = 0.5
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_P_AWARE_LOSS = True
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_P_AWARE_MARGIN = 0.2
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_P_AWARE_WEIGHT = 0.1
_C.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_SPO_AGNOSTIC_COMPENSATION = False

_C.MODEL.ROI_RELATION_HEAD.IMP_FEATURE_UPDATE_STEP = 0
_C.MODEL.ROI_RELATION_HEAD.MSDN_FEATURE_UPDATE_STEP = 0
_C.MODEL.ROI_RELATION_HEAD.GRCNN_FEATURE_UPDATE_STEP = 0
_C.MODEL.ROI_RELATION_HEAD.GRCNN_SCORE_UPDATE_STEP = 0

# -----------------------------------------------------------------------------
# Test Options
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.OUTPUT_RELATION_FEATURE = False
_C.TEST.OUTPUT_ATTRIBUTE_FEATURE = False
