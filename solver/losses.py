from pkg import *
from config import *
from helpers import *

# ============================================================================
# Loss Functions
# ============================================================================


@partial(jit, static_argnums=(4,))
def loss_data_dsm(params_net, params_phys, x_batch, t_batch, config, key):
    D = params_phys["D"]
    sigma_t = jnp.sqrt(config.gamma_noise * D * config.dt_obs)
    key, subkey = random.split(key)
    eps = random.normal(subkey, shape=x_batch.shape)
    x_tilde = x_batch + sigma_t * eps
    s_pred = vmap(score_network, in_axes=(None, 0, 0))(
        params_net, x_tilde, t_batch)
    s_target = -eps / sigma_t
    loss = 0.001 * jnp.mean((s_pred - s_target)**2)
    return loss


@partial(jit, static_argnums=(4, 5))
def loss_physics_residual(params_net, params_phys, x_colloc, t_colloc, config, drift_fn, key):
    def residual_single(x, t, subkey):
        dt_s = grad(lambda t_: jnp.sum(score_network(params_net, x, t_)))(t)
        def G_scalar_fn(x_): return jnp.sum(compute_G_operator(params_net, params_phys, x_, t, subkey, drift_fn, config))
        grad_G = grad(G_scalar_fn)(x)
        residual = dt_s + grad_G
        return residual**2
    keys = random.split(key, len(x_colloc))
    residuals = vmap(residual_single)(x_colloc, t_colloc, keys)
    return jnp.mean(residuals)


def dsm_guide(params_phys, true_params, phys_keys, D_to_sigma=False):
    l2 = 0.0
    for k in phys_keys:
        if k == "D" and D_to_sigma:
            est = jnp.sqrt(params_phys[k])
            tru = true_params["sigma"]
        else:
            est = params_phys[k]
            if k == "D":
                tru = true_params["sigma"] ** 2
            else:
                tru = true_params[k]
        l2 += (est - tru) ** 2
    return l2 / len(phys_keys)


def total_loss(params_net, params_phys, data_batch, colloc_batch, config, drift_fn, key, true_params=None, sde_name=None):
    key_data, key_phys = random.split(key)
    x_data, t_data = data_batch
    x_colloc, t_colloc = colloc_batch
    L_data = loss_data_dsm(params_net, params_phys, x_data, t_data, config, key_data)
    L_phys = loss_physics_residual(params_net, params_phys, x_colloc, t_colloc, config, drift_fn, key_phys)
    L_total = config.lambda_data * L_data + config.lambda_physics * L_phys
    L_guidance = 0.0
    if true_params is not None:
        phys_keys = list(params_phys.keys())
        L_guidance = dsm_guide(params_phys, true_params, phys_keys, D_to_sigma=True)
        L_total = L_total + config.lambda_guidance * L_guidance
    return L_total, {"L_data": L_data, "L_phys": L_phys, "L_total": L_total, "L_guidance": L_guidance}
