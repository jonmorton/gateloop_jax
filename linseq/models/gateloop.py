from dataclasses import dataclass

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp

from ..config import ModelConfig
from .common import MLP, LayerNorm

ortho_init = jax.nn.initializers.orthogonal()


@ModelConfig.register_subclass("gateloop")
@dataclass
class GateLoopConfig(ModelConfig):
    hdim: int = 512
    num_layers: int = 12

    @property
    def model_class(self):
        return GateLoop


def gate_loop_operator(k, v, q, a):
    kv = (k @ v).astype(a.dtype)

    def binary_operator(e_i, e_j):
        a_i, kv_i = e_i
        a_j, kv_j = e_j
        return a_j * a_i, a_j * kv_i + kv_j

    _, y = jax.lax.associative_scan(binary_operator, (a, kv), axis=1)

    y = jnp.real(y).astype(k.dtype)

    return q @ y


class GateLoopLayer(eqx.Module):
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wa: jax.Array
    wg: jax.Array
    wo: jax.Array
    gn: nn.GroupNorm
    bias: jax.Array

    def __init__(self, key, dim, seq_len):
        """
        q - query
        k - key
        v - value
        a - state transition
        g - gating with silu activation
        o - output
        """

        q_key, k_key, v_key, a_key, g_key, o_key = jax.random.split(key, 6)

        self.wq = ortho_init(q_key, (dim, dim))
        self.wk = ortho_init(k_key, (dim, dim))
        self.wv = ortho_init(v_key, (dim, dim))
        self.wa = ortho_init(a_key, (dim, dim * 2))
        self.wg = ortho_init(g_key, (dim, dim))
        self.wo = ortho_init(o_key, (dim, dim))
        self.gn = nn.GroupNorm(
            64, dim
        )  # no idea what group size original author used..
        self.bias = jnp.zeros(dim)

    def __call__(self, x):
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        a = x @ self.wa
        g = x @ self.wg

        # complex state transition
        a = a.astype(jnp.float32)
        a_real, a_imag = jnp.split(a, 2, axis=-1)
        a_complex = jax.nn.sigmoid(a_real) * jnp.exp(1j * a_imag)

        # associative scan
        y = gate_loop_operator(k, v, q, a_complex)

        y = jax.vmap(self.gn)(y)
        y = y * jax.nn.silu(g)
        o = y @ self.wo

        return o + self.bias


class Block(eqx.Module):
    attn: GateLoopLayer
    mlp: MLP
    norm1: LayerNorm
    norm2: LayerNorm

    def __init__(self, key, dim, seqlen):
        k1, k2 = jax.random.split(key)
        self.attn = GateLoopLayer(k1, dim, seqlen)
        self.mlp = MLP(k2, dim)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def __call__(self, x, key=None):
        # x = jax.vmap(self.norm1)(x + self.attn(x))
        # x = jax.vmap(self.norm2)(x + jax.vmap(self.mlp)(x))
        x = x + self.attn(jax.vmap(self.norm1)(x))
        x = x + jax.vmap(self.mlp)(jax.vmap(self.norm2)(x))
        return x


class GateLoop(eqx.Module):
    tok_embed: nn.Embedding
    blocks: nn.Sequential
    final_norm: nn.LayerNorm
    final_proj: jax.Array

    def __init__(self, cfg: GateLoopConfig, key=None):
        if key is None:
            key = jax.random.PRNGKey(42)
        k1, k2, *kx = jax.random.split(key, 2 + cfg.num_layers)
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hdim, key=k1)
        self.final_norm = LayerNorm(cfg.hdim)
        self.final_proj = ortho_init(k2, (cfg.hdim, cfg.vocab_size))
        self.blocks = nn.Sequential([Block(k, cfg.hdim, cfg.seq_len) for k in kx])

    def __call__(self, x):
        x = jax.vmap(self.tok_embed)(x)
        x = self.blocks(x)
        x = jax.vmap(self.final_norm)(x)
        x = x @ self.final_proj
        return x
