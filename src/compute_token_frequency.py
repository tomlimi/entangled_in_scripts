import logging
import os
import shutil
import argparse
import sys
import json
from tqdm import tqdm
from collections import OrderedDict

from transformers import XLMRobertaTokenizerFast

logging.basicConfig(level=logging.INFO)


def get_tokenizer_path(tokenizer_dir, tokenizer_type, lang, alpha, NV):
    return os.path.join(tokenizer_dir, tokenizer_type, lang, f"alpha-{alpha}_N-{NV}")


# getting tokenizer
def get_tokenizer(tokenizer_dir, tokenizer_type, lang, alpha, NV):
    tokenizer_path = get_tokenizer_path(tokenizer_dir, tokenizer_type, lang, alpha, NV)
    logging.info(f"Loading tokenizer from {tokenizer_path}")
    if not os.path.exists(tokenizer_path):
        raise ValueError(f"Tokenizer not found at {tokenizer_path}")
    return (
        XLMRobertaTokenizerFast.from_pretrained(tokenizer_path, unk_token="<unk>"),
        tokenizer_path,
    )


def save_token_frequency(tokens_with_freq, decoded_tokens_with_freq, out_path, name):
    """Function to save token frequencies and log arguments to a file"""

    # copy current script to the output directory
    shutil.copyfile(sys.argv[0], os.path.join(out_path, "{name}_script.py"))
    # save the arguments
    with open(os.path.join(out_path, "{name}_args.txt"), "w") as log_file:
        log_file.write(" ".join(sys.argv[1:]))

    for save_name, save_object in [
        (f"{name}.json", tokens_with_freq),
        (f"{name}_decoded.json", decoded_tokens_with_freq),
    ]:
        save_path = os.path.join(out_path, save_name)
        with open(save_path, "w", encoding="utf-8") as outfile:
            logging.info(f"Writing frequencies to {save_path}")
            json.dump(
                OrderedDict(save_object),
                outfile,
                indent=2,
                ensure_ascii=False,
            )


def batch(iterator, batch_size):
    """Yield elements from iterator in batches of batch_size."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def main(args):
    alpha = args.alpha
    vocab_size = args.vocab_size
    languages = args.languages
    languages_str = "-".join(languages)
    data_paths = args.data_list
    lowercase = not args.cased

    type = args.type
    out_dir = args.out_dir

    # load the tokenizer
    tokenizer, tokenizer_path = get_tokenizer(
        out_dir, type, languages_str, alpha, vocab_size
    )

    # open the train data
    batch_size = 10000

    counter = {token_id: 0 for token_id in tokenizer.get_vocab().values()}
    for data_path in data_paths:
        logging.info(f"Reading lines from {data_path}")
        with open(data_path, "r") as f:
            # go through the file line by line in batches
            # NOTE: we strip the newline character from the end of each line
            for line_batch in tqdm(batch(map(lambda s: s.rstrip(), f), batch_size)):
                for tokenized_line in tokenizer(line_batch)["input_ids"]:
                    for id in tokenized_line:
                        counter[id] += 1

    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
    tokens_with_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    decoded_tokens_with_freq = [
        (id_to_token[token_id], freq) for token_id, freq in tokens_with_freq
    ]

    save_token_frequency(
        tokens_with_freq, decoded_tokens_with_freq, tokenizer_path, args.name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_list", nargs="+", help="<Required> Set flag", required=True
    )
    parser.add_argument("-o", "--out_dir", type=str, required=True)
    parser.add_argument(
        "-a", "--alpha", type=str, required=True, help="Balancing coefficient alpha."
    )
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        required=True,
        help="List of languages the tokenizer was trained on.",
    )
    parser.add_argument("-v", "--vocab_size", type=int, required=True)
    parser.add_argument("-t", "--type", type=str, required=False, default="unigram")
    parser.add_argument(
        "-n", "--name", type=str, required=False, default="token_frequencies"
    )
    parser.add_argument("-c", "--cased", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
