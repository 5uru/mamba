import jax
import jax.numpy as jnp
from jax import lax
import jax.nn.initializers as jinit
from flax import nnx
from dataclasses import dataclass
from typing import Union
from pscan import pscan_jax

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
    dt_init_floor = 1e-4
    d_inner: int = 128
    bias: bool = True
    conv_bias: bool = True


class MambaBlock(nnx.Module):

    def __init__(self,  config: MambaConfig, rngs: nnx.Rngs):

        self.config = config

        self.in_proj = nnx.Linear(config.d_model,  2* config.d_inner,  use_bias=config.bias, rngs=rngs)
        self.conv1d = nnx.Conv(in_features=config.d_inner,
                               out_features=config.d_inner,
                               kernel_size=(config.d_conv,), # Doit être un tuple en Flax
                               strides=1,
                               padding=((config.d_conv - 1, 0),), # Voir l'explication ci-dessous
                               feature_group_count=config.d_inner, # Équivalent de 'groups' dans PyTorch
                               use_bias=config.conv_bias,
                               rngs=rngs)
        self.x_proj = nnx.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, use_bias=False, rngs=rngs)


        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        kernel_init = jinit.uniform(dt_init_std)



        def init_dt_bias(config):
            def init(key, shape, dtype):
                # Étape 1 : Générer dt réparti uniformément dans l'espace logarithmique
                log_dt_min = lax.log(config.dt_min)
                log_dt_max = lax.log(config.dt_max)

                # torch.rand -> jax.random.uniform
                u = jax.random.uniform(key, shape, dtype=dtype)
                dt = jnp.exp(u * (log_dt_max - log_dt_min) + log_dt_min)

                # .clamp(min=...) -> jnp.maximum(...)
                dt = jnp.maximum(dt, config.dt_init_floor)

                # Étape 2 : Calculer l'inverse de la softplus
                # inv_dt = dt + log(-expm1(-dt))
                inv_dt = dt + jnp.log(-jnp.expm1(-dt))

                return inv_dt.astype(dtype)
            return init

        self.dt_proj = nnx.Linear(
                in_features=config.dt_rank,
                out_features=config.d_inner,
                use_bias=True,
                kernel_init=kernel_init,        # Initialisation des poids
                bias_init=init_dt_bias(config), # Initialisation spéciale du biais
                rngs=rngs
        )

        A_1d = jnp.arange(1, config.d_state + 1, dtype=jnp.float32)
        A = jnp.tile(A_1d, (config.d_inner, 1))
        self.A_log = nnx.Param(jnp.log(A))

        self.D = nnx.Param(jnp.ones((config.d_inner, config.d_state), dtype=jnp.float32))

        self.out_proj = nnx.Linear(config.d_inner, config.d_model, use_bias=config.bias, rngs=rngs)


    def selective_scan(self, x, delta, A, B, C, D):

        deltaA = lax.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        hs = pscan_jax(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y

    

