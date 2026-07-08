import os
import json
import math
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from sentencepiece import SentencePieceProcessor
import sentencepiece as spm

from mamba import MambaConfig, RMSNorm, ResidualBlock


# ==============================================================================
# TOKENIZER
# ==============================================================================

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


# ==============================================================================
# MODEL
# ==============================================================================

class MambaEncoder(nnx.Module):
    def __init__(self, config: MambaConfig, rngs: nnx.Rngs):
        self.layers = [ResidualBlock(config, rngs=rngs) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MambaDecoder(nnx.Module):
    def __init__(self, config: MambaConfig, rngs: nnx.Rngs):
        self.layers = [ResidualBlock(config, rngs=rngs) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MambaTranslationModel(nnx.Module):
    def __init__(self, config: MambaConfig, src_vocab: int, tgt_vocab: int, rngs: nnx.Rngs):
        self.src_embedding = nnx.Embed(src_vocab, config.d_model, rngs=rngs)
        self.tgt_embedding = nnx.Embed(tgt_vocab, config.d_model, rngs=rngs)
        self.encoder = MambaEncoder(config, rngs=rngs)
        self.decoder = MambaDecoder(config, rngs=rngs)
        self.lm_head = nnx.Linear(config.d_model, tgt_vocab, rngs=rngs)
        self.cross_proj = nnx.Linear(config.d_model * 2, config.d_model, rngs=rngs)
        self.config = config

    def encode(self, src_tokens: jnp.ndarray) -> jnp.ndarray:
        x = self.src_embedding(src_tokens)
        return self.encoder(x)

    def __call__(self, src_tokens: jnp.ndarray, tgt_tokens: jnp.ndarray) -> jnp.ndarray:
        enc_out = self.encode(src_tokens)
        tgt_emb = self.tgt_embedding(tgt_tokens)
        tgt_len = tgt_tokens.shape[1]

        cross_input = jnp.concatenate([
            jnp.tile(enc_out[:, :1, :], (1, tgt_len, 1)),
            tgt_emb
        ], axis=-1)
        cross_input = self.cross_proj(cross_input)

        dec_out = self.decoder(cross_input)
        logits = self.lm_head(dec_out)
        return logits


# ==============================================================================
# DATA PIPELINE
# ==============================================================================

def collate_fn(src_batch, tgt_batch, src_tokenizer, tgt_tokenizer, max_src_len: int = 128, max_tgt_len: int = 128):
    src_ids = [src_tokenizer.encode(t)[:max_src_len] for t in src_batch]
    tgt_ids = [tgt_tokenizer.encode(t)[:max_tgt_len] for t in tgt_batch]

    src_pad_len = max(len(ids) for ids in src_ids)
    tgt_pad_len = max(len(ids) for ids in tgt_ids)

    src_arr = jnp.array([ids + [src_tokenizer.pad_id] * (src_pad_len - len(ids)) for ids in src_ids])
    tgt_arr = jnp.array([ids + [tgt_tokenizer.pad_id] * (tgt_pad_len - len(ids)) for ids in tgt_ids])

    tgt_input = tgt_arr[:, :-1]
    tgt_label = tgt_arr[:, 1:]

    return src_arr, tgt_input, tgt_label


def create_batches(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, batch_size, max_src_len, max_tgt_len, key):
    n = len(src_texts)
    indices = jax.random.permutation(key, n)

    for start in range(0, n, batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_src = [src_texts[int(i)] for i in batch_indices]
        batch_tgt = [tgt_texts[int(i)] for i in batch_indices]
        yield collate_fn(batch_src, batch_tgt, src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len)


# ==============================================================================
# LEARNING RATE SCHEDULE
# ==============================================================================

def get_cosine_schedule_with_warmup(warmup_steps: int, total_steps: int, max_lr: float, min_lr: float):
    def schedule(step):
        step = jnp.array(step, dtype=jnp.float32)
        warmup = jnp.array(warmup_steps, dtype=jnp.float32)
        total = jnp.array(total_steps, dtype=jnp.float32)

        lr = jnp.where(
            step < warmup,
            max_lr * step / warmup,
            min_lr + (max_lr - min_lr) * 0.5 * (1 + jnp.cos(jnp.pi * (step - warmup) / (total - warmup)))
        )
        return jnp.where(step >= total, min_lr, lr)

    return schedule


# ==============================================================================
# TRAINING
# ==============================================================================

@nnx.jit
def train_step(model, optimizer, src, tgt_input, tgt_label):
    def loss_fn(model):
        logits = model(src, tgt_input)
        one_hot = jax.nn.one_hot(tgt_label, logits.shape[-1])
        mask = (tgt_label != 0).astype(jnp.float32)
        loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


@nnx.jit
def eval_step(model, src, tgt_input, tgt_label):
    logits = model(src, tgt_input)
    one_hot = jax.nn.one_hot(tgt_label, logits.shape[-1])
    mask = (tgt_label != 0).astype(jnp.float32)
    loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    loss = (loss * mask).sum() / mask.sum()

    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == tgt_label).astype(jnp.float32)
    acc = (correct * mask).sum() / mask.sum()
    return loss, acc


def greedy_decode(model, src_tokens: jnp.ndarray, tgt_tokenizer, max_len: int = 100) -> list[int]:
    decoder_input = jnp.array([[tgt_tokenizer.bos_id()]], dtype=jnp.int32)

    for _ in range(max_len):
        logits = model(src_tokens[None, :], decoder_input)
        next_token = int(jnp.argmax(logits[:, -1, :], axis=-1))
        if next_token == tgt_tokenizer.eos_id:
            break
        decoder_input = jnp.concatenate([decoder_input, jnp.array([[next_token]])], axis=1)

    return decoder_input[0].tolist()


def compute_bleu(references: list[str], hypotheses: list[str]) -> float:
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs = [[ref.split()] for ref in references]
        hyps = [hyp.split() for hyp in hypotheses]
        smoothing = SmoothingFunction().method1
        return corpus_bleu(refs, hyps, smoothing_function=smoothing) * 100
    except Exception:
        return 0.0


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Mamba for Fon -> French Translation")
    parser.add_argument("--data_path", type=str, default="ffr_dataset_fon_fr_without_diacritics.txt")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dt_rank", type=int, default=32)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand_factor", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_src_len", type=int, default=128)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")
    print(f"Data: {args.data_path} (Fon -> French)")

    print("Loading parallel data...")
    src_texts, tgt_texts = load_parallel_data(args.data_path)
    total = len(src_texts)
    print(f"Total pairs: {total}")

    val_size = int(total * args.val_ratio)
    train_size = total - val_size
    train_src, val_src = src_texts[:train_size], src_texts[train_size:]
    train_tgt, val_tgt = tgt_texts[:train_size], tgt_texts[train_size:]
    print(f"Train: {train_size} | Val: {val_size}")

    src_spm_path = save_dir / f"src_spm_{args.vocab_size}.model"
    tgt_spm_path = save_dir / f"tgt_spm_{args.vocab_size}.model"

    if not src_spm_path.exists():
        print("Training tokenizers...")
        train_tokenizers(args.data_path, str(save_dir), args.vocab_size)
        print("Done.")

    src_tokenizer = SPMTokenizer(str(src_spm_path))
    tgt_tokenizer = SPMTokenizer(str(tgt_spm_path))
    print(f"Src vocab: {len(src_tokenizer)} | Tgt vocab: {len(tgt_tokenizer)}")

    config = MambaConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dt_rank=args.dt_rank,
        d_state=args.d_state,
        expand_factor=args.expand_factor,
    )

    rngs = nnx.Rngs(args.seed)
    model = MambaTranslationModel(
        config=config,
        src_vocab=len(src_tokenizer),
        tgt_vocab=len(tgt_tokenizer),
        rngs=rngs,
    )

    total_steps = args.epochs * train_size // args.batch_size
    lr_schedule = get_cosine_schedule_with_warmup(args.warmup_steps, total_steps, args.max_lr, args.min_lr)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=0.1),
    )
    optimizer = nnx.Optimizer(model, tx)

    param_count = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
    print(f"Model parameters: {param_count / 1e6:.2f}M")
    print(f"Total training steps: {total_steps}")

    global_step = 0
    best_val_loss = float("inf")
    train_key = jax.random.PRNGKey(args.seed)

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        train_key, subkey = jax.random.split(train_key)

        for src_batch, tgt_input_batch, tgt_label_batch in create_batches(
            train_src, train_tgt, src_tokenizer, tgt_tokenizer,
            args.batch_size, args.max_src_len, args.max_tgt_len, subkey
        ):
            loss = train_step(model, optimizer, src_batch, tgt_input_batch, tgt_label_batch)
            epoch_loss += float(loss)
            n_batches += 1
            global_step += 1

            if global_step % args.log_interval == 0:
                lr = float(lr_schedule(global_step))
                avg_loss = epoch_loss / n_batches
                print(f"  Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

            if global_step % args.eval_interval == 0:
                model.eval()
                val_losses, val_accs = [], []
                val_key = jax.random.PRNGKey(999)
                for v_src, v_tgt_in, v_tgt_label in create_batches(
                    val_src, val_tgt, src_tokenizer, tgt_tokenizer,
                    args.batch_size, args.max_src_len, args.max_tgt_len, val_key
                ):
                    vl, va = eval_step(model, v_src, v_tgt_in, v_tgt_label)
                    val_losses.append(float(vl))
                    val_accs.append(float(va))

                avg_vl = sum(val_losses) / len(val_losses)
                avg_va = sum(val_accs) / len(val_accs)
                print(f"  [VAL] Loss: {avg_vl:.4f} | Acc: {avg_va:.4f}")

                if avg_vl < best_val_loss:
                    best_val_loss = avg_vl
                    save_checkpoint(model, global_step, config, save_dir / "best_model")
                    print(f"  -> Saved best model")

                model.train()

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"\nEpoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.4f}")

        save_checkpoint(model, global_step, config, save_dir / f"checkpoint_epoch{epoch + 1}")

        print(f"\nSample translations (epoch {epoch + 1}):")
        model.eval()
        for i in range(min(5, val_size)):
            src_text = val_src[i]
            tgt_text = val_tgt[i]
            src_tokens = jnp.array([src_tokenizer.encode(src_text)], dtype=jnp.int32)
            decoded_ids = greedy_decode(model, src_tokens[0], tgt_tokenizer, max_len=args.max_tgt_len)
            pred_text = tgt_tokenizer.decode(decoded_ids)
            print(f"  SRC: {src_text}")
            print(f"  TGT: {tgt_text}")
            print(f"  PRED: {pred_text}")
            print()

    print("\nTraining complete!")
    save_checkpoint(model, global_step, config, save_dir / "final_model")

    print("\nRunning final BLEU on validation set...")
    model.eval()
    references, hypotheses = [], []
    val_key = jax.random.PRNGKey(999)
    for v_src_batch, v_tgt_in, v_tgt_label in create_batches(
        val_src, val_tgt, src_tokenizer, tgt_tokenizer,
        args.batch_size, args.max_src_len, args.max_tgt_len, val_key
    ):
        for j in range(v_src_batch.shape[0]):
            src_t = val_src[j]
            tgt_t = val_tgt[j]
            decoded_ids = greedy_decode(model, v_src_batch[j], tgt_tokenizer, max_len=args.max_tgt_len)
            pred_t = tgt_tokenizer.decode(decoded_ids)
            references.append(tgt_t)
            hypotheses.append(pred_t)

    bleu = compute_bleu(references, hypotheses)
    print(f"Final BLEU Score: {bleu:.2f}")


def save_checkpoint(model, step: int, config, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "config.json", "w") as f:
        json.dump({
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "dt_rank": config.dt_rank,
            "d_state": config.d_state,
            "expand_factor": config.expand_factor,
            "d_conv": config.d_conv,
            "step": step,
        }, f, indent=2)
    print(f"  Checkpoint saved to {path}")


if __name__ == "__main__":
    main()
