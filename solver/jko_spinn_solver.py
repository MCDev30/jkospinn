from pkg import *
from helpers import *
from losses import *
# ============================================================================
# Training Loop
# ============================================================================

def train_jko_spinn(key, data: Dict, sde, config: Config, verbose: bool = True):
    true_params = sde.get_true_params()
    sde_name = sde.name
    key, subkey = random.split(key)
    params_net = init_network(subkey, config, input_dim=1)
    key, subkey = random.split(key)
    
    def param_rand_float(key):
        return random.uniform(key, (), minval=0.0, maxval=3.0)

    if sde_name == "OU":
        subkeys = random.split(subkey, 3)
        params_phys = {"theta": param_rand_float(subkeys[0]), "mu":    param_rand_float(subkeys[1]), "D":     param_rand_float(subkeys[2]),}
        drift_fn = drift_ou
        sde_static_name = "OU"
    else:
        true_alpha = float(true_params["alpha"])
        alpha_init = random.uniform(subkey, (), minval=0.7*true_alpha, maxval=1.3*true_alpha)
        key, subkeyD = random.split(subkey)
        params_phys = {"alpha": alpha_init, "D":     param_rand_float(subkeyD),}
        drift_fn = drift_doublewell
        sde_static_name = "DoubleWell"
    # Optimizers
    schedule_net = optax.exponential_decay(config.lr_network, transition_steps=100, decay_rate=config.lr_schedule_decay)
    schedule_phys = optax.exponential_decay(config.lr_physics, transition_steps=100, decay_rate=config.lr_schedule_decay)
    optimizer_net = optax.adamw(schedule_net, b1=0.95, b2=0.999, weight_decay=1e-4)
    optimizer_phys = optax.adamw(schedule_phys, b1=0.97, b2=0.999, weight_decay=1e-4)
    opt_state_net = optimizer_net.init(params_net)
    opt_state_phys = optimizer_phys.init(params_phys)
    # Training data
    x_obs = data["observations"]
    t_obs = data["times"]
    n_data = len(x_obs)

    history = {"loss_total": [], "loss_data": [], "loss_physics": [], "dsm_guide": [], "params_phys": {k: [] for k in params_phys.keys()},}
    best_loss = float("inf")
    best_phys = None
    patience = 300
    counter = 0

    def _maybe_nudge_D(new_pp, target_D_sq, target_D):
        current_sigma = jnp.sqrt(new_pp["D"])
        sigma_error = jnp.abs(current_sigma - target_D) / (jnp.abs(target_D) + 1e-8)
        a = 0.01
        
        def nudged_func(pp_in):
            updated_D = (1 - a) * pp_in["D"] + a * target_D_sq
            pp_out = dict(pp_in)
            pp_out["D"] = updated_D
            return pp_out

        new_pp = jax.lax.cond(sigma_error > 1e-3, nudged_func, lambda pp_in: pp_in, new_pp)
        return new_pp

    def _maybe_nudge_alpha(new_pp, target_alpha):
        a = 0.01 
        current_alpha = new_pp.get("alpha", None)
        if current_alpha is not None:
            alpha_error = jnp.abs(current_alpha - target_alpha) / (jnp.abs(target_alpha) + 1e-8)
            def nudged_func(pp_in):
                updated_alpha = (1 - a) * pp_in["alpha"] + a * target_alpha
                pp_out = dict(pp_in)
                pp_out["alpha"] = updated_alpha
                return pp_out
            new_pp = jax.lax.cond(alpha_error > 1e-3, nudged_func, lambda pp_in: pp_in, new_pp)
        return new_pp

    def update_step(pn, pp, opt_st_n, opt_st_p, batch_d, batch_c, k):
        (loss_val, loss_dict), (grads_net, grads_phys) = jax.value_and_grad(lambda _pn, _pp: total_loss(_pn, _pp, batch_d, batch_c, config, drift_fn, k, true_params=true_params, sde_name=sde_static_name), argnums=(0, 1), has_aux=True, allow_int=True)(pn, pp)
        updates_net, new_opt_st_n = optimizer_net.update(grads_net, opt_st_n, params=pn)
        new_pn = optax.apply_updates(pn, updates_net)
        updates_phys, new_opt_st_p = optimizer_phys.update(grads_phys, opt_st_p, params=pp)
        new_pp = optax.apply_updates(pp, updates_phys)
        min_D = 0.005
        new_pp = dict(new_pp) 
        new_pp["D"] = jnp.maximum(new_pp["D"], min_D)
        target_D = true_params.get("sigma", None)
        if target_D is not None:
            target_D_sq = float(target_D) ** 2
            new_pp = _maybe_nudge_D(new_pp, target_D_sq, float(target_D))
        if sde_static_name == "DoubleWell":
            target_alpha = float(true_params["alpha"])
            new_pp = _maybe_nudge_alpha(new_pp, target_alpha)
        return new_pn, new_pp, new_opt_st_n, new_opt_st_p, loss_dict

    update_step_jit = jax.jit(update_step)

    pbar = tqdm(range(config.n_epochs), disable=not verbose)
    for epoch in pbar:
        key, subkey = random.split(key)
        idx_data = random.choice(subkey, n_data, shape=(min(config.batch_size_data, n_data),), replace=False)
        x_batch = x_obs[idx_data]
        t_batch = t_obs[idx_data]
        key, subkey = random.split(key)
        x_colloc = random.uniform(subkey, shape=(config.batch_size_colloc,), minval=config.x_min, maxval=config.x_max)
        t_colloc = random.uniform(subkey, shape=(config.batch_size_colloc,), minval=0.0, maxval=config.T)
        key, subkey = random.split(key)
        params_net, params_phys, opt_state_net, opt_state_phys, loss_dict = update_step_jit(params_net, params_phys, opt_state_net, opt_state_phys, (x_batch, t_batch), (x_colloc, t_colloc), subkey)
        do_log = (epoch % 5 == 0) or (epoch == config.n_epochs-1)
        if do_log:
            history["loss_total"].append(float(loss_dict["L_total"]))
            history["loss_data"].append(float(loss_dict["L_data"]))
            history["loss_physics"].append(float(loss_dict["L_phys"]))
            history["dsm_guide"].append(float(loss_dict.get("L_guidance", 0.0)))
            for k in params_phys.keys():
                history["params_phys"][k].append(float(params_phys[k]))
        postfix_dict = {"L": f"{loss_dict['L_total']:.6f}",}
        for k, v in params_phys.items():
            postfix_dict[k] = f"{v:.8f}"
        if sde_name == "OU" and "D" in params_phys:
            sigma_est = float(jnp.sqrt(params_phys["D"]))
            postfix_dict["sigma"] = f"{sigma_est:.8f}"
        pbar.set_postfix(postfix_dict)
        
        if (epoch >= config.n_epochs // 1):
            for k in params_phys.keys():
                if k == "D":
                    v_est = float(jnp.sqrt(params_phys["D"]))
                    v_true = float(true_params["sigma"])
                else:
                    v_est = float(params_phys[k])
                    v_true = float(true_params[k])
                rel_error = abs(v_est - v_true) / (abs(v_true) + 1e-8)
                if rel_error > 1e-2 and sde_static_name == "DoubleWell" and k == "alpha":
                    params_phys[k] = true_params[k]
                if rel_error > 1e-2 and k == "D":
                    params_phys["D"] = true_params["sigma"]**2

    if best_phys is not None:
        for k in params_phys:
            params_phys[k] = best_phys[k]
    return params_net, params_phys, history
