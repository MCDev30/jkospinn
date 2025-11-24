from pkg import *

# ============================================================================
# Evaluation
# ============================================================================

def compute_relative_error(estimated, true):
    return jnp.abs(estimated - true) / (jnp.abs(true) + 1e-10)


def evaluate_results(params_phys, sde, history):
    true_params = sde.get_true_params()
    results = {}
    for key in params_phys.keys():
        if key == "D":
            est_sigma = jnp.sqrt(params_phys["D"])
            true_sigma = true_params["sigma"]
            results["sigma"] = {
                "estimated": float(est_sigma),
                "true": true_sigma,
                "relative_error": float(compute_relative_error(est_sigma, true_sigma))
            }
        else:
            results[key] = {
                "estimated": float(params_phys[key]),
                "true": true_params[key],
                "relative_error": float(compute_relative_error(params_phys[key], true_params[key]))
            }
    return results
