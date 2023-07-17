#!/bin/bash

export OMP_NUM_THREADS=1
export gpu_num=4
export CUDA_VISIBLE_DEVICES="4,5,6,7"

predictor=SquatPredictor
archive_dir="checkpoints/sgcls"
model="sgcls.pth"

python -m torch.distributed.launch --master_port 12029 --nproc_per_node=$gpu_num  \
  tools/relation_test_net.py \
  --config-file "$archive_dir/config.yml" \
   OUTPUT_DIR $archive_dir \
   TEST.IMS_PER_BATCH $[$gpu_num] \
   MODEL.WEIGHT  "$archive_dir/$model"\
   MODEL.ROI_RELATION_HEAD.PREDICTOR $predictor \
   MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.RHO 0.5 \
   MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.BETA 1.0 \