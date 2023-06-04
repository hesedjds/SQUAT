export gpu_num=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python -m torch.distributed.launch --master_port 11228 --nproc_per_node=$gpu_num \
tools/relation_train_net.py \
--config-file "checkpoints/predcls/config.yml" \
OUTPUT_DIR 'checkpoints' \
EXPERIMENT_NAME 'predcls1' 