import pickle
import math
import argparse
import glob
import logging
from pathlib import Path
from tqdm import tqdm
import unicodedata

from transformers import BertTokenizer, RobertaTokenizer
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.tokenization_minilm import MinilmTokenizer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'minilm': MinilmTokenizer,
    'roberta': RobertaTokenizer,
    'unilm': UnilmTokenizer,
}


def read_traces_from_file(file_name):
    with open(file_name, "rb") as fin:
        meta = pickle.load(fin)
        num_samples = meta["num_samples"]
        samples = []
        for _ in range(num_samples):
            samples.append(pickle.load(fin))
    return samples


def get_best_sequence(sample, eos_id, pad_id, length_penalty=None, alpha=None, expect=None, min_len=None):
    # if not any((length_penalty, alpha, expect, min_len)):
    #     raise ValueError(
    #         "You can only specify length penalty or alpha, but not both.")
    scores = sample["scores"]
    wids_list = sample["wids"]
    ptrs = sample["ptrs"]

    last_frame_id = len(scores) - 1
    for i, wids in enumerate(wids_list):
        if all(wid in (eos_id, pad_id) for wid in wids):
            last_frame_id = i
            break
    while all(wid == pad_id for wid in wids_list[last_frame_id]):
        last_frame_id -= 1

    max_score = -math.inf
    frame_id = -1
    pos_in_frame = -1

    for fid in range(last_frame_id + 1):
        for i, wid in enumerate(wids_list[fid]):
            if fid <= last_frame_id and scores[fid][i] >= 0:
                # skip paddings
                continue
            if (wid in (eos_id, pad_id)) or fid == last_frame_id:
                s = scores[fid][i]
                if length_penalty:
                    if expect:
                        s -= length_penalty * math.fabs(fid+1 - expect)
                    else:
                        s += length_penalty * (fid + 1)
                elif alpha:
                    s = s / math.pow((5 + fid + 1) / 6.0, alpha)
                if s > max_score:
                    # if (frame_id != -1) and min_len and (fid+1 < min_len):
                    #     continue
                    max_score = s
                    frame_id = fid
                    pos_in_frame = i
    if frame_id == -1:
        seq = []
    else:
        seq = [wids_list[frame_id][pos_in_frame]]
        for fid in range(frame_id, 0, -1):
            pos_in_frame = ptrs[fid][pos_in_frame]
            seq.append(wids_list[fid - 1][pos_in_frame])
        seq.reverse()
    return seq


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def simple_postprocess(tk_list):
    # truncate duplicate punctuations
    while tk_list and len(tk_list) > 4 and len(tk_list[-1]) == 1 and unicodedata.category(tk_list[-1]).startswith('P') and all(it == tk_list[-1] for it in tk_list[-4:]):
        tk_list = tk_list[:-3]
    return tk_list


# def include_unk(line):
#     return " UNK ".join(line.split('<unk>')).strip()


def main(args):
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.tokenizer_name, do_lower_case=args.do_lower_case, 
        cache_dir=args.cache_dir if args.cache_dir else None)
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token

    eos_id, pad_id = set(tokenizer.convert_tokens_to_ids([eos_token, pad_token]))
    logger.info("*********************************************")
    logger.info(" EOS TOKEN = {}, ID = {}".format(eos_token, eos_id))
    logger.info(" PAD TOKEN = {}, ID = {}".format(pad_token, pad_id))
    logger.info("*********************************************")

    for input_file in tqdm(glob.glob(args.input)):
        if not Path(input_file+'.trace.pickle').exists():
            continue
        print(input_file)
        samples = read_traces_from_file(input_file+'.trace.pickle')

        results = []

        for s in samples:
            word_ids = get_best_sequence(s, eos_id, pad_id, alpha=args.alpha,
                                         length_penalty=args.length_penalty, expect=args.expect, min_len=args.min_len)
            tokens = tokenizer.convert_ids_to_tokens(word_ids)
            buf = []
            for t in tokens:
                if t in (eos_token, pad_token):
                    break
                else:
                    buf.append(t)
            if args.model_type == "roberta":
                output_text = " ".join(simple_postprocess(tokenizer.convert_tokens_to_string(buf).split(' ')))
                if '\n' in output_text:
                    output_text = " [S_SEPX_SEP] ".join(output_text.split('\n'))
            else:
                output_text = " ".join(simple_postprocess(detokenize(buf)))

            results.append(output_text)

        fn_out = input_file + '.'
        if args.length_penalty:
            fn_out += 'lenp'+str(args.length_penalty)
        if args.expect:
            fn_out += 'exp'+str(args.expect)
        if args.alpha:
            fn_out += 'alp'+str(args.alpha)
        if args.min_len:
            fn_out += 'minl'+str(args.min_len)
        with open(fn_out, "w", encoding="utf-8") as fout:
            for line in results:
                fout.write(line)
                fout.write("\n")
        logger.info("Output file = [%s]" % fn_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(TOKENIZER_CLASSES.keys()))
    parser.add_argument("--alpha", default=None, type=float)
    parser.add_argument("--length_penalty", default=None, type=float)
    parser.add_argument("--expect", default=None, type=float,
                        help="Expectation of target length.")
    parser.add_argument("--min_len", default=None, type=int)
    # tokenizer_name
    parser.add_argument("--tokenizer_name", default=None, type=str, required=True, 
                        help="tokenizer name")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    args = parser.parse_args()

    main(args)
