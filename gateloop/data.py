import os

import numpy as np
import sentencepiece


def data_iter(path, batch_size, chunk_size):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ss = batch_size * chunk_size
    n_item = data.size // ss

    while True:
        for i in range(n_item):
            chunk = data[i * ss : i * ss + ss].reshape(batch_size, chunk_size)
            yield chunk


class DatasetManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=os.path.join(base_dir, "vocab.model")
        )

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    @property
    def doc_separator(self):
        return self.tokenizer.eos_id()

    def train_iter(self, batch_size, chunk_size):
        path = os.path.join(self.base_dir, "train.bin")
        yield from data_iter(path, batch_size, chunk_size)

    def validation_iter(self, batch_size, chunk_size):
        path = os.path.join(self.base_dir, "validation.bin")
        yield from data_iter(path, batch_size, chunk_size)
