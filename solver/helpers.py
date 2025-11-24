from pkg import *
from config import *

# ============================================================================
# Neural Network: Score Function s_phi(x, t)
# ============================================================================

def fourier_features(x, t, B, scale=1.0):
    v = jnp.concatenate([x.reshape(-1, 1), t.reshape(-1, 1)], axis=-1)
    proj = 2 * jnp.pi * scale * (v @ B.T)
    return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


def init_network(key, config: Config, input_dim: int = 1):
    keys = random.split(key, len(config.hidden_dims) + 2)
    B = random.normal(keys[0], (config.fourier_features, input_dim + 1)) * config.fourier_scale
    layers = []
    in_dim = 2 * config.fourier_features
    for i, out_dim in enumerate(config.hidden_dims):
        W = random.normal(keys[i+1], (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
        b = jnp.zeros(out_dim)
        layers.append((W, b))
        in_dim = out_dim
    W_out = random.normal(keys[-1], (in_dim, input_dim)) * jnp.sqrt(2.0 / in_dim)
    b_out = jnp.zeros(input_dim)
    layers.append((W_out, b_out))
    return {"B": B, "layers": layers}


def swish(x):
    return x * jax.nn.sigmoid(x)


def score_network(params, x, t):
    x = jnp.atleast_1d(x)
    t = jnp.atleast_1d(t)
    is_batched = x.ndim > 1
    if not is_batched:
        x = x[None, :]
        t = t[None]
    h = fourier_features(x, t, params["B"], scale=1.0)
    for i, (W, b) in enumerate(params["layers"][:-1]):
        h = h @ W + b
        h = swish(h)
    W_out, b_out = params["layers"][-1]
    out = h @ W_out + b_out
    if not is_batched:
        out = out[0]
    return out

# ============================================================================
# Physics Operators
# ============================================================================

def drift_ou(x, theta, mu):
    return -theta * (x - mu)


def drift_doublewell(x, alpha):
    return -4 * alpha * x * (x**2 - 1)


def hutchinson_trace_estimator(key, fn, x, D, n_samples=1):
    x_arr = jnp.atleast_1d(x)
    d = x_arr.shape[0]

    def single_sample(v):
        def jvp_fn(x_in):
            val = fn(x_in)
            val = jnp.atleast_1d(val)
            return jnp.dot(val, v)
        grad_fn = grad(lambda x_: jnp.sum(jvp_fn(x_)))
        hvp = grad_fn(x)
        return jnp.dot(v, D * hvp)
    keys = random.split(key, n_samples)
    v_samples = random.normal(keys[0], (n_samples, d))
    traces = vmap(single_sample)(v_samples)
    return jnp.mean(traces)


def compute_G_operator(params_net, params_phys, x, t, key, drift_fn, config):
    s = score_network(params_net, x, t)
    if "theta" in params_phys:
        F = drift_fn(x, params_phys["theta"], params_phys["mu"])
    else:
        F = drift_fn(x, params_phys["alpha"])
    D = params_phys["D"]
    div_F = grad(lambda x: jnp.sum(drift_fn(x, *[params_phys[k] for k in params_phys if k != "D"])))(x)
    s_dot_F = jnp.dot(s, F)
    s_D_s = jnp.dot(s, D * s)
    trace_term = hutchinson_trace_estimator(key, lambda x_: score_network(params_net, x_, t),  x,  D,  n_samples=config.n_hutchinson_samples)
    G = div_F + s_dot_F - 0.5 * s_D_s - 0.5 * trace_term
    return G
