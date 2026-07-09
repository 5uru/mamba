import jax
import jax.numpy as jnp


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
