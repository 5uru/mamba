import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from mamba import MambaConfig
from tokenizer import SPMTokenizer, load_parallel_data, train_tokenizers
from model import MambaTranslationModel
from data import create_batches
from train_utils import (
    get_cosine_schedule_with_warmup,
    train_step,
    eval_step,
    greedy_decode,
    compute_bleu,
    save_checkpoint,
)
import metrics

# ==============================================================================
# CONFIG — edit these values directly
# ==============================================================================

CONFIG = {
    "data_path": "ffr_dataset_v2.txt",
    "d_model": 256,
    "n_layers": 4,
    "dt_rank": 32,
    "d_state": 16,
    "expand_factor": 2,
    "batch_size": 16,
    "epochs": 20,
    "max_lr": 3e-4,
    "min_lr": 1e-5,
    "warmup_steps": 500,
    "max_src_len": 128,
    "max_tgt_len": 128,
    "val_ratio": 0.05,
    "save_dir": "checkpoints",
    "vocab_size": 8000,
    "log_interval": 50,
    "eval_interval": 500,
    "seed": 42,
}


def main():
    cfg = CONFIG
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")
    print(f"Data: {cfg['data_path']} (Fon -> French)")

    print("Loading parallel data...")
    src_texts, tgt_texts = load_parallel_data(cfg["data_path"])
    total = len(src_texts)
    print(f"Total pairs: {total}")

    val_size = int(total * cfg["val_ratio"])
    train_size = total - val_size
    train_src, val_src = src_texts[:train_size], src_texts[train_size:]
    train_tgt, val_tgt = tgt_texts[:train_size], tgt_texts[train_size:]
    print(f"Train: {train_size} | Val: {val_size}")

    src_spm_path = save_dir / f"src_spm_{cfg['vocab_size']}.model"
    tgt_spm_path = save_dir / f"tgt_spm_{cfg['vocab_size']}.model"

    if not src_spm_path.exists():
        print("Training tokenizers...")
        train_tokenizers(cfg["data_path"], str(save_dir), cfg["vocab_size"])
        print("Done.")

    src_tokenizer = SPMTokenizer(str(src_spm_path))
    tgt_tokenizer = SPMTokenizer(str(tgt_spm_path))
    print(f"Src vocab: {len(src_tokenizer)} | Tgt vocab: {len(tgt_tokenizer)}")

    config = MambaConfig(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        dt_rank=cfg["dt_rank"],
        d_state=cfg["d_state"],
        expand_factor=cfg["expand_factor"],
    )

    rngs = nnx.Rngs(cfg["seed"])
    model = MambaTranslationModel(
        config=config,
        src_vocab=len(src_tokenizer),
        tgt_vocab=len(tgt_tokenizer),
        rngs=rngs,
    )

    total_steps = cfg["epochs"] * train_size // cfg["batch_size"]
    lr_schedule = get_cosine_schedule_with_warmup(cfg["warmup_steps"], total_steps, cfg["max_lr"], cfg["min_lr"])

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=0.1),
    )
    optimizer = nnx.Optimizer(model, tx)

    param_count = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
    print(f"Model parameters: {param_count / 1e6:.2f}M")
    print(f"Total training steps: {total_steps}")

    metrics.init_run(cfg, save_dir=cfg["save_dir"])
    print(f"Metrics file: {cfg['save_dir']}/metrics.jsonl")

    global_step = 0
    best_val_loss = float("inf")
    train_key = jax.random.PRNGKey(cfg["seed"])

    for epoch in range(cfg["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{cfg['epochs']}")
        print(f"{'='*60}")

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        train_key, subkey = jax.random.split(train_key)

        epoch_start = time.time()
        for src_batch, tgt_input_batch, tgt_label_batch in create_batches(
            train_src, train_tgt, src_tokenizer, tgt_tokenizer,
            cfg["batch_size"], cfg["max_src_len"], cfg["max_tgt_len"], subkey
        ):
            step_start = time.time()
            loss = train_step(model, optimizer, src_batch, tgt_input_batch, tgt_label_batch)
            step_time = time.time() - step_start
            epoch_loss += float(loss)
            n_batches += 1
            global_step += 1

            if global_step % cfg["log_interval"] == 0:
                lr = float(lr_schedule(global_step))
                avg_loss = epoch_loss / n_batches
                print(f"  Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
                metrics.log_metrics(global_step, {
                    "train_loss": avg_loss,
                    "learning_rate": lr,
                    "epoch": epoch + 1,
                }, step_time=step_time)

            if global_step % cfg["eval_interval"] == 0:
                model.eval()
                val_losses, val_accs = [], []
                val_key = jax.random.PRNGKey(999)
                for v_src, v_tgt_in, v_tgt_label in create_batches(
                    val_src, val_tgt, src_tokenizer, tgt_tokenizer,
                    cfg["batch_size"], cfg["max_src_len"], cfg["max_tgt_len"], val_key
                ):
                    vl, va = eval_step(model, v_src, v_tgt_in, v_tgt_label)
                    val_losses.append(float(vl))
                    val_accs.append(float(va))

                avg_vl = sum(val_losses) / len(val_losses)
                avg_va = sum(val_accs) / len(val_accs)
                print(f"  [VAL] Loss: {avg_vl:.4f} | Acc: {avg_va:.4f}")
                metrics.log_metrics(global_step, {
                    "val_loss": avg_vl,
                    "val_accuracy": avg_va,
                    "best_val_loss": best_val_loss,
                })

                if avg_vl < best_val_loss:
                    best_val_loss = avg_vl
                    save_checkpoint(model, global_step, config, save_dir / "best_model")
                    print(f"  -> Saved best model")

                model.train()

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"\nEpoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.1f}s")
        metrics.log_metrics(global_step, {
            "epoch_loss": avg_epoch_loss,
            "epoch_time_s": epoch_time,
            "epoch": epoch + 1,
        })

        save_checkpoint(model, global_step, config, save_dir / f"checkpoint_epoch{epoch + 1}")

        print(f"\nSample translations (epoch {epoch + 1}):")
        model.eval()
        for i in range(min(5, val_size)):
            src_text = val_src[i]
            tgt_text = val_tgt[i]
            src_tokens = jnp.array([src_tokenizer.encode(src_text)], dtype=jnp.int32)
            decoded_ids = greedy_decode(model, src_tokens[0], tgt_tokenizer, max_len=cfg["max_tgt_len"])
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
        cfg["batch_size"], cfg["max_src_len"], cfg["max_tgt_len"], val_key
    ):
        for j in range(v_src_batch.shape[0]):
            src_t = val_src[j]
            tgt_t = val_tgt[j]
            decoded_ids = greedy_decode(model, v_src_batch[j], tgt_tokenizer, max_len=cfg["max_tgt_len"])
            pred_t = tgt_tokenizer.decode(decoded_ids)
            references.append(src_t)
            hypotheses.append(pred_t)

    bleu = compute_bleu(references, hypotheses)
    print(f"Final BLEU Score: {bleu:.2f}")
    metrics.log_metrics(global_step, {"final_bleu": bleu})
    print(f"\nMetrics saved to: {cfg['save_dir']}/metrics.jsonl")


if __name__ == "__main__":
    main()
