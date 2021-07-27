# path of the fine-tuned checkpoint
DIR=/dnn/sheng.s/dissUnilm/
kd_weight=0.6
num_training_steps=45000

MODEL_PATH=${DIR}distill_checkpoints_kd${kd_weight}_step${num_training_steps}/ckpt-${num_training_steps}/
#SPLIT=dev
SPLIT=test
# input file that you would like to decode
INPUT_JSON=${DIR}cnndm.${SPLIT}.uncased_tokenized.json

export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py \
  --fp16 --model_type unilm --tokenizer_name unilm1.2-base-uncased --do_lower_case --input_file ${INPUT_JSON} --split $SPLIT \
  --model_path ${MODEL_PATH} --max_seq_length 768 --max_tgt_length 160 --batch_size 32 --beam_size 5 \
  --length_penalty 0.7 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "." --min_len 48

#SPLIT=dev
GOLD_PATH=${DIR}${SPLIT}.target
# ${MODEL_PATH}.${SPLIT} is the predicted target file
python evaluations/eval_for_cnndm.py --pred ${MODEL_PATH}.${SPLIT} --gold ${GOLD_PATH} --split ${SPLIT} --trunc_len 160 --perl
