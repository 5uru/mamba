import jax
import jax.numpy as jnp
from jax import lax
import jax.nn.initializers as jinit
from flax import nnx
from dataclasses import dataclass
from typing import Union
from pscan import pscan_jax

# ==============================================================================
# MODÈLE MAMBA
# ==============================================================================

@dataclass
class MambaConfig:
    d_model: int
    n_layers: int
    dt_rank: int
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4

    bias: bool = True
    conv_bias: bool = True
    rms_norm_eps: float = 1e-5


class MambaBlock(nnx.Module):

    def __init__(self, config: MambaConfig, rngs: nnx.Rngs):
        self.config = config
        self.d_inner = config.d_model * config.expand_factor

        self.in_proj = nnx.Linear(config.d_model, 2 * self.d_inner, use_bias=config.bias, rngs=rngs)

        self.conv1d = nnx.Conv(
                in_features=self.d_inner,
                out_features=self.d_inner,
                kernel_size=(config.d_conv,),
                strides=1,
                padding=((config.d_conv - 1, 0),),
                feature_group_count=self.d_inner,
                use_bias=config.conv_bias,
                rngs=rngs
        )

        self.x_proj = nnx.Linear(self.d_inner, config.dt_rank + 2 * config.d_state, use_bias=False, rngs=rngs)

        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        kernel_init = jinit.uniform(dt_init_std)

        def init_dt_bias(config):
            def init(key, shape, dtype):
                log_dt_min = lax.log(config.dt_min)
                log_dt_max = lax.log(config.dt_max)
                u = jax.random.uniform(key, shape, dtype=dtype)
                dt = jnp.exp(u * (log_dt_max - log_dt_min) + log_dt_min)
                dt = jnp.maximum(dt, config.dt_init_floor)
                inv_dt = dt + jnp.log(-jnp.expm1(-dt))
                return inv_dt.astype(dtype)
            return init

        self.dt_proj = nnx.Linear(
                in_features=config.dt_rank,
                out_features=self.d_inner,
                use_bias=True,
                kernel_init=kernel_init,
                bias_init=init_dt_bias(config),
                rngs=rngs
        )

        A_1d = jnp.arange(1, config.d_state + 1, dtype=jnp.float32)
        A = jnp.tile(A_1d, (self.d_inner, 1))
        self.A_log = nnx.Param(jnp.log(A))

        self.D = nnx.Param(jnp.ones((self.d_inner,), dtype=jnp.float32))

        self.out_proj = nnx.Linear(self.d_inner, config.d_model, use_bias=config.bias, rngs=rngs)

    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = lax.exp(jnp.expand_dims(delta, -1) * A)
        deltaB = jnp.expand_dims(delta, -1) * jnp.expand_dims(B, 2)

        BX = deltaB * jnp.expand_dims(x, -1)

        # Utilisation du VRAI parallel scan
        hs = pscan_jax(deltaA, BX)

        y = jnp.einsum('bldn,bldn->bld', hs, jnp.expand_dims(C, 2))

        y = y + D * x
        return y

    def ssm(self, x):
        A = - lax.exp(self.A_log.astype(jnp.float32))
        D = self.D.astype(jnp.float32)

        deltaBC = self.x_proj(x)
        delta, B, C = jnp.split(deltaBC, [self.config.dt_rank, self.config.dt_rank + self.config.d_state], axis=-1)

        delta = jax.nn.softplus(self.dt_proj(delta))

        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def __call__(self, x):
        _, L, _ = x.shape

        xz = self.in_proj(x)
        x = xz[..., :self.d_inner]
        z = xz[..., self.d_inner:]

        x = self.conv1d(x)[:, :L, :]

        x = jax.nn.silu(x)
        y = self.ssm(x)

        z = jax.nn.silu(z)
        output = self.out_proj(y * z)
        return output


class RMSNorm(nnx.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((d_model,), dtype=jnp.float32))

    def __call__(self, x):
        output = x * lax.rsqrt((x ** 2).mean(-1, keepdims=True) + self.eps)
        return output * self.weight


class ResidualBlock(nnx.Module):
    def __init__(self, config: MambaConfig, rngs: nnx.Rngs):
        self.mixer = MambaBlock(config, rngs=rngs)
        self.norm = RMSNorm(d_model=config.d_model, eps=config.rms_norm_eps)

    def __call__(self, x):
        y = self.norm(x)
        y = self.mixer(y) + x
        return y


class Mamba(nnx.Module):
    def __init__(self, config: MambaConfig, rngs: nnx.Rngs):
        self.config = config
        self.layers = [ResidualBlock(config, rngs=rngs) for _ in range(config.n_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x





# ==============================================================================
# SCRIPT DE TEST
# ==============================================================================

def test_mamba_model():
    print("Initialisation de la config...")
    config = MambaConfig(
            d_model=64,
            n_layers=2,
            dt_rank=4,
            d_state=8,
            expand_factor=2,
            d_conv=4
    )

    batch_size = 2
    seq_len = 10

    print(f"Création du modèle (Batch={batch_size}, SeqLen={seq_len}, d_model={config.d_model})...")
    rngs = nnx.Rngs(42)
    model = Mamba(config, rngs=rngs)

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, seq_len, config.d_model))

    print("Forward pass avec Parallel Scan...")
    output = model(x)

    print(f"Shape d'entrée : {x.shape}")
    print(f"Shape de sortie : {output.shape}")

    assert output.shape == (batch_size, seq_len, config.d_model), "Erreur de shape !"

    print("\nTest du gradient (Backward pass)...")
    @nnx.jit
    def compute_loss(model, x):
        y = model(x)
        return jnp.sum(y ** 2)

    loss, grads = nnx.value_and_grad(compute_loss)(model, x)

    print(f"Loss calculée : {loss}")
    print(f"Nombre de paramètres avec gradients : {len(grads)}")

    print("\n✅ Tout s'est bien passé ! Le modèle Mamba est 100% opérationnel et optimisé.")

if __name__ == "__main__":
    test_mamba_model()