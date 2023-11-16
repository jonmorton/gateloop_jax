from dataclasses import dataclass

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp

nn.MultiheadAttention

from ..config import ModelConfig
from .common import MLP, LayerNorm, ortho_init


@ModelConfig.register_subclass("gpt2")
@dataclass
class GPT2Config(ModelConfig):
    hdim: int = 512
    num_layers: int = 12

    @property
    def model_class(self):
        return GPT2


class CausalSelfAttention(eqx.Module):
    nhead: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    Wqkv: jax.Array
    Wout: jax.Array
    bias: jax.Array
    causal_mask: jax.Array

    def __init__(self, key, dim, seq_len, dim_head=64):
        self.nhead = dim // dim_head
        self.dim_head = dim_head
        k1, k2 = jax.random.split(key)
        self.Wqkv = ortho_init(k1, (dim, dim * 3), jnp.float32)
        self.Wout = ortho_init(k2, (dim, dim), jnp.float32)
        self.bias = jnp.zeros(dim)
        self.causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, x):
        seq_len = x.shape[-2]
        qkv = x @ self.Wqkv
        qkv = jnp.reshape(qkv, (-1, 3, self.nhead, self.dim_head))
        qkv = jnp.swapaxes(qkv, 0, 2)  # (nhead, 3, seq, dim_head)
        q, k, v = [jax.lax.index_in_dim(qkv, i, -3, keepdims=False) for i in range(3)]

        logits = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(self.dim_head)
        mask = jax.lax.stop_gradient(self.causal_mask[:seq_len, :seq_len])
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
        dtype = logits.dtype
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(dtype)
        attn_outs = probs @ v
        attn_outs = jnp.reshape(
            jnp.swapaxes(attn_outs, 0, 1), (-1, self.nhead * self.dim_head)
        )

        proj_out = attn_outs @ self.Wout
        return proj_out + self.bias


class Block(eqx.Module):
    attn: CausalSelfAttention
    mlp: MLP
    norm1: LayerNorm
    norm2: LayerNorm

    def __init__(self, key, dim, seqlen):
        k1, k2 = jax.random.split(key)
        self.attn = CausalSelfAttention(k1, dim, seqlen)
        self.mlp = MLP(k2, dim)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def __call__(self, x, key=None):
        x = x + self.attn(jax.vmap(self.norm1)(x))
        x = x + self.mlp(jax.vmap(self.norm2)(x))
        return x


class GPT2(eqx.Module):
    tok_embed: nn.Embedding
    pos_embed: jax.Array
    blocks: nn.Sequential
    final_norm: nn.LayerNorm
    final_proj: jax.Array

    def __init__(self, cfg: GPT2Config, key=None):
        if key is None:
            key = jax.random.PRNGKey(42)
        k1, k2, *kx = jax.random.split(key, 2 + cfg.num_layers)
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hdim, key=k1)
        self.pos_embed = jnp.zeros((cfg.seq_len, cfg.hdim), jnp.float32)
        self.final_norm = LayerNorm(cfg.hdim)
        self.final_proj = ortho_init(k2, (cfg.hdim, cfg.vocab_size))
        self.blocks = nn.Sequential([Block(k, cfg.hdim, cfg.seq_len) for k in kx])

    def __call__(self, x):
        x = jax.vmap(self.tok_embed)(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = jax.vmap(self.final_norm)(x)
        x = x @ self.final_proj
        return x
