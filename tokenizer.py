import os

from sentencepiece import SentencePieceProcessor
import sentencepiece as spm


class SPMTokenizer:
    def __init__(self, model_path: str):
        self.sp = SentencePieceProcessor()
        self.sp.Load(model_path)
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        tokens = self.sp.EncodeAsIds(text.strip())
        if add_special:
            tokens = [self.bos_id] + tokens + [self.eos_id]
        return tokens

    def decode(self, ids: list[int]) -> str:
        return self.sp.DecodeIds(ids)

    def __len__(self):
        return self.sp.GetPieceSize()


def load_parallel_data(data_path: str):
    src_texts, tgt_texts = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                src_texts.append(parts[0])
                tgt_texts.append(parts[1])
    return src_texts, tgt_texts


def train_tokenizers(data_path: str, save_dir: str, vocab_size: int = 8000):
    src_texts, tgt_texts = load_parallel_data(data_path)

    src_file = os.path.join(save_dir, "src_corpus.txt")
    tgt_file = os.path.join(save_dir, "tgt_corpus.txt")

    with open(src_file, "w") as f:
        for t in src_texts:
            f.write(t + "\n")

    with open(tgt_file, "w") as f:
        for t in tgt_texts:
            f.write(t + "\n")

    for corpus_file, prefix in [(src_file, "src"), (tgt_file, "tgt")]:
        model_prefix = os.path.join(save_dir, f"{prefix}_spm_{vocab_size}")
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )

    os.remove(src_file)
    os.remove(tgt_file)
    print(f"Tokenizers trained: {vocab_size} vocab each")
