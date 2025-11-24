from pkg import *
from sde_simulator import *

def generate_data(key, sde: SDESimulator, config: Config) -> Dict:
    n_steps_traj = int(config.T / config.dt_sim)
    subsample_rate = max(1, int(config.dt_obs / config.dt_sim))
    trajectories, observations, times = [], [], []
    for i in range(config.n_trajectories):
        key, subkey = random.split(key)
        if sde.name == "OU":
            x0 = random.normal(subkey, shape=(1,)) * 0.5
        else:
            x0 = random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(1,))
            x0 = x0 + random.normal(subkey, shape=(1,)) * 0.1
        traj = SDESimulator.euler_maruyama(
            subkey, x0, sde.drift, sde.sigma, config.dt_sim, n_steps_traj)
        obs = traj[::subsample_rate]
        t_obs = jnp.arange(len(obs)) * config.dt_obs
        trajectories.append(traj)
        observations.append(obs)
        times.append(t_obs)
    all_obs = jnp.concatenate(observations)
    all_times = jnp.concatenate(times)
    return {
        "observations": all_obs,
        "times": all_times,
        "trajectories": trajectories,
        "n_trajectories": config.n_trajectories,
        "dt_obs": config.dt_obs,
    }
