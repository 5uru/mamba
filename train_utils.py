import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx


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
