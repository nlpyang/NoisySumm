# path of training data
DIR=/dnn/sheng.s/dissUnilm/
kd_weight=0.6
num_training_steps=60000
export TEACHER_MODEL=${DIR}yang_cnndm_unilmv1.2.pt
export TRAIN_FILE=${DIR}cnndm.train.uncased_tokenized.json
# folder used to save fine-tuned checkpoints
export OUTPUT_DIR=${DIR}distill_checkpoints_kd${kd_weight}_step${num_training_steps}_real
# folder used to cache package dependencies
export CACHE_DIR=${DIR}transformer_package_cache


#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 run_seq2seq.py \
  --train_file $TRAIN_FILE --output_dir $OUTPUT_DIR \
  --model_type unilm --model_name_or_path unilm1.2-base-uncased --do_lower_case --fp16 --fp16_opt_level O2 \
  --max_source_seq_length 608 --max_target_seq_length 160 --per_gpu_train_batch_size 8 --gradient_accumulation_steps 2 \
  --learning_rate 7e-5 --num_warmup_steps 1000 --num_training_steps $num_training_steps --cache_dir $CACHE_DIR --save_steps 1500 \
  --use_distill 1 --kd_weight $kd_weight --teacher_model $TEACHER_MODEL --teacher_dropout 1 --min_lr 0
