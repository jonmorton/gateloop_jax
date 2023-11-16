import os
import random

import numpy as np
import sentencepiece
from datasets import load_dataset
from tqdm import tqdm

NUM_PROC = 12
VOCAB_SIZE = 8192
TRAIN_MAXLEN = 2048
TRAIN_MAXDOC = 2000000


def tokenizer_train_iter(dataset):
    i = 0
    for item in dataset.shuffle():
        text = item["text"].strip()
        if len(text) < 10:
            continue
        if len(text) > TRAIN_MAXLEN:
            offset = random.randint(0, len(text) - TRAIN_MAXLEN - 1)
            text = text[offset : offset + TRAIN_MAXLEN]
        yield text
        i += 1
        if i >= TRAIN_MAXDOC:
            break


if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(__file__), "wiki")
    os.makedirs(root_dir, exist_ok=True)

    dataset = load_dataset("wikipedia", "20220301.en", num_proc=NUM_PROC)

    # sentencepiece.SentencePieceTrainer.train(
    #     sentence_iterator=tokenizer_train_iter(dataset["train"]),
    #     vocab_size=VOCAB_SIZE,
    #     model_prefix=os.path.join(root_dir, "vocab"),
    #     model_type="bpe",
    #     byte_fallback=True,
    #     max_sentence_length=TRAIN_MAXLEN * 4 // 3,
    #     num_threads=os.cpu_count(),
    # )

    spm = sentencepiece.SentencePieceProcessor(
        model_file=os.path.join(root_dir, "vocab.model")
    )

    dataset = dataset["train"].train_test_split(test_size=0.03)
    dataset["validation"] = dataset.pop("test")

    def process(example):
        tokens = spm.encode(example["text"])
        tokens.append(spm.eos_id())
        out = {"tokens": tokens, "len": len(tokens)}
        return out

    # tokenize the dataset
    processed = dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=NUM_PROC,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in processed.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"split {split} has {arr_len:,} tokens")
        filename = os.path.join(root_dir, f"{split}.bin")
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["tokens"]).astype(dtype)
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
