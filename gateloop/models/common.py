import equinox as eqx
import jax
import jax.numpy as jnp

ortho_init = jax.nn.initializers.orthogonal()


class LayerNorm(eqx.Module):
    eps: float = eqx.field(static=True)
    use_weight: bool = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    weight: jax.Array
    bias: jax.Array

    def __init__(
        self,
        dim,
        eps: float = 1e-5,
        use_weight: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.weight = jnp.ones(dim) if use_weight else None
        self.bias = jnp.zeros(dim) if use_bias else None

    def __call__(
        self,
        x,
        key=None,
    ):
        mean = jnp.mean(x, keepdims=True, axis=-1)
        variance = jnp.var(x.astype(jnp.float32), keepdims=True, axis=-1)
        variance = jnp.maximum(0.0, variance)
        inv = jax.lax.rsqrt(variance + self.eps).astype(x.dtype)
        out = (x - mean) * inv
        if self.use_weight:
            out = self.weight * out
        if self.use_bias:
            out = out + self.bias
        return out


class MLP(eqx.Module):
    w_up: jax.Array
    w_down: jax.Array
    bias: jax.Array

    def __init__(self, key, dim, expand=4):
        k1, k2 = jax.random.split(key)
        self.w_up = ortho_init(k1, (dim, dim * expand), jnp.float32)
        self.w_down = ortho_init(k2, (dim * expand, dim), jnp.float32)
        self.bias = jnp.zeros(dim)

    def __call__(self, x):
        x = x @ self.w_up
        x = jax.nn.gelu(x)
        x = x @ self.w_down
        x = x + self.bias
        return x
