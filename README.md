# Noisy Self-Knowledge Distillation for Text Summarization
Codes for NAACL 2021 paper 'Noisy Self-Knowledge Distillation for Text Summarization'

The code is based on UNILM, and summarization data can be download at (https://github.com/microsoft/unilm/tree/master/s2s-ft)




## Train teacher model

MODEL_PATH=../models/xsum.unilm/ckpt-40000
SPLIT=test
INPUT_JSON=../data/xsum.test.uncased_tokenized.json

export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py \
  --fp16 --model_type unilm --tokenizer_name unilm1.2-base-uncased --input_file ${INPUT_JSON} --split $SPLIT --do_lower_case \
  --model_path ${MODEL_PATH} --max_seq_length 512 --max_tgt_length 48 --batch_size 32 --beam_size 5 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "."
  
## Distill a student model

TRAIN_FILE=../data/xsum.train.uncased_tokenized.json
CACHE_DIR=../../cache
OUTPUT_DIR=../models/xsum.unilm.distill
TEACHER=../models/xsum.unilm/ckpt-40000/pytorch_model.bin

BATCH_SIZE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8  --master_port 29886  run_seq2seq.py \
  --train_file $TRAIN_FILE --output_dir $OUTPUT_DIR \
  --model_type unilm --model_name_or_path unilm1.2-base-uncased --do_lower_case --fp16 --fp16_opt_level O2 \
  --max_source_seq_length 464  --max_target_seq_length 48  --per_gpu_train_batch_size $BATCH_SIZE --gradient_accumulation_steps 1 \
  --learning_rate 7e-5 --num_warmup_steps 500  --num_training_steps 40000  --cache_dir $CACHE_DIR --save_steps 2000 \
  --use_distill 1 --kd_weight 0.6  --teacher_dropout_prob  0.15  --use_teacher_dropout  1 --teacher_model $TEACHER   --word_drop_prob 0.1 --use_noisy_student 1 --sent_shuffle_k 2  

## Decode
MODEL_PATH=../models/xsum.unilm.distill/ckpt-40000
SPLIT=test
INPUT_JSON=../data/xsum.test.uncased_tokenized.json

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py --model_type unilm --tokenizer_name unilm1.2-base-uncased --input_file ${INPUT_JSON} --split $SPLIT --do_lower_case \
  --model_path ${MODEL_PATH} --max_seq_length 512 --max_tgt_length 48 --batch_size 32 --beam_size 8 \
  --length_penalty 0.9 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "." --min_len 5


