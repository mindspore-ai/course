"""Network parameters."""
from easydict import EasyDict
pretrain_config = EasyDict({
    # LR
    "base_lr": 0.0009,
    "warmup_step": 30000,
    "warmup_ratio": 1/3.0,
    "total_epoch": 10,
})
finetune_config = EasyDict({
    # LR
    "base_lr": 0.0005,
    "warmup_step": 300,
    "warmup_ratio": 1/3.0,
    "total_epoch": 50,
})

# use for low case number
config = EasyDict({
    "img_width": 960,
    "img_height": 576,
    "keep_ratio": False,
    "flip_ratio": 0.0,
    "photo_ratio": 0.0,
    "expand_ratio": 1.0,

    # anchor
    "feature_shapes": (36, 60),
    "num_anchors": 14,
    "anchor_base": 16,
    "anchor_height": [2, 4, 7, 11, 16, 23, 33, 48, 68, 97, 139, 198, 283, 406],
    "anchor_width": [16],

    # rpn
    "rpn_in_channels": 256,
    "rpn_feat_channels": 512,
    "rpn_loss_cls_weight": 1.0,
    "rpn_loss_reg_weight": 3.0,
    "rpn_cls_out_channels": 2,

    # bbox_assign_sampler
    "neg_iou_thr": 0.5,
    "pos_iou_thr": 0.7,
    "min_pos_iou": 0.001,
    "num_bboxes": 30240,
    "num_gts": 256,
    "num_expected_neg": 512,
    "num_expected_pos": 256,

    #proposal
    "activate_num_classes": 2,
    "use_sigmoid_cls": False,

    # train proposal
    "rpn_proposal_nms_across_levels": False,
    "rpn_proposal_nms_pre": 2000,
    "rpn_proposal_nms_post": 1000,
    "rpn_proposal_max_num": 1000,
    "rpn_proposal_nms_thr": 0.7,
    "rpn_proposal_min_bbox_size": 8,

    # rnn structure
    "input_size": 512,
    "num_step": 60,
    "rnn_batch_size": 36,
    "hidden_size": 128,

    # training
    "warmup_mode": "linear",
    "batch_size": 1,
    "momentum": 0.9,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 2,
    "keep_checkpoint_max": 5,
    "save_checkpoint_path": "/home/ma-user/work/output/checkpoint",
    "use_dropout": False,
    "loss_scale": 1,
    "weight_decay": 1e-4,

    # test proposal
    "rpn_nms_pre": 2000,
    "rpn_nms_post": 1000,
    "rpn_max_num": 1000,
    "rpn_nms_thr": 0.7,
    "rpn_min_bbox_min_size": 8,
    "test_iou_thr": 0.7,
    "test_max_per_img": 100,
    "test_batch_size": 1,
    "use_python_proposal": False,

    # text proposal connection
    "max_horizontal_gap": 60,
    "text_proposals_min_scores": 0.7,
    "text_proposals_nms_thresh": 0.2,
    "min_v_overlaps": 0.7,
    "min_size_sim": 0.7,
    "min_ratio": 0.5,
    "line_min_score": 0.9,
    "text_proposals_width": 16,
    "min_num_proposals": 2,

    # create dataset
    "hwdb_pretrain_path": ['../dataset/detection/pretrain_images','../dataset/detection/pretrain_labels'],
    "hwdb_finetune_path": ['../dataset/detection/finetune_images','../dataset/detection/finetune_labels'],
    "hwdb_test_path": ['../dataset/detection/test_images','../dataset/detection/test_labels'],
    "pretrain_dataset_path": "../dataset/detection/ms_pretrain_dataset",
    "finetune_dataset_path": "../dataset/detection/ms_finetune_dataset",
    "test_dataset_path": "../dataset/detection/ms_test_dataset",

    # training dataset
    "pretraining_dataset_file": "../dataset/detection/ms_pretrain_dataset/ctpn_pretrain.mindrecord0",
    "finetune_dataset_file": "../dataset/detection/ms_finetune_dataset/ctpn_finetune.mindrecord0"
})
