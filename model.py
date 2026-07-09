import jax.numpy as jnp
from flax import nnx

from mamba import MambaConfig, RMSNorm, ResidualBlock


class MambaEncoder(nnx.Module):
    """
    Mamba Encoder module.
    """
    def __init__(self, config: MambaConfig, rngs: nnx.Rngs):
        self.layers = [ResidualBlock(config, rngs=rngs) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MambaDecoder(nnx.Module):
    """
    Mamba Decoder module.
    """
    def __init__(self, config: MambaConfig, rngs: nnx.Rngs):
        self.layers = [ResidualBlock(config, rngs=rngs) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MambaTranslationModel(nnx.Module):
    """
    Mamba Translation Model module.
    """
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
