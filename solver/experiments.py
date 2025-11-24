from pkg import *
from jko_spinn_solver import *
from eval_ import *

# ============================================================================
# Multi-Initialization Experiment (for Double-Well)
# ============================================================================
def multi_init_experiment(key, data, sde, config, n_runs=2):  
    results_all = []
    print(f"\n{'='*60}")
    print(f"Multi-Initialization Experiment: {n_runs} runs")
    print(f"{'='*60}\n")
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...")
        key, subkey = random.split(key)
        params_net, params_phys, history = train_jko_spinn(subkey, data, sde, config, verbose=False)
        results = evaluate_results(params_phys, sde, history)
        results_all.append(results)
        for param_name in results.keys():
            est = results[param_name]["estimated"]
            true = results[param_name]["true"]
            re = results[param_name]["relative_error"] * 100
            print(f"  {param_name}: {est:.6f} (true: {true:.6f}, RE: {re:.4f}%)")
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    for param_name in results_all[0].keys():
        estimates = [r[param_name]["estimated"] for r in results_all]
        true_val = results_all[0][param_name]["true"]
        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        mean_re = np.mean([r[param_name]["relative_error"] for r in results_all]) * 100
        print(f"\n{param_name}:")
        print(f"  True value:    {true_val:.6f}")
        print(f"  Mean estimate: {mean_est:.6f} ± {std_est:.6f}")
        print(f"  Mean RE:       {mean_re:.4f}%")
        print(
            f"  Min/Max:       {np.min(estimates):.6f} / {np.max(estimates):.6f}")
    return results_all


# ============================================================================
# Additional Visualization Functions
# ============================================================================

def plot_multi_init_results(results_all, sde, save_path=None):
    fig, axes = plt.subplots(1, len(results_all[0].keys()), figsize=(18, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    param_names = list(results_all[0].keys())
    true_params = sde.get_true_params()
    for idx, param_name in enumerate(param_names):
        ax = axes[idx]
        estimates = [r[param_name]["estimated"] for r in results_all]
        true_val = true_params.get(param_name, true_params.get("sigma"))
        ax.hist(estimates, bins=max(3, len(results_all)//2), alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(true_val, color='red', linestyle='--', linewidth=2, label='True value')
        ax.axvline(np.mean(estimates), color='green', linestyle='-', linewidth=2, label='Mean estimate')
        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        ax.set_xlabel(f'{param_name}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{param_name}: μ={mean_est:.6f}, σ={std_est:.6f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Multi-Initialization Results ({len(results_all)} runs)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sparsity_results(results_sparsity, sde, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_traj_values = [r["n_traj"] for r in results_sparsity]
    param_names = list(results_sparsity[0]["results"].keys())
    ax = axes[0]
    for param_name in param_names:
        errors = [r["results"][param_name]["relative_error"] * 100 for r in results_sparsity]
        ax.plot(n_traj_values, errors, marker='o', linewidth=2, label=param_name, alpha=0.7)
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Parameter Estimation vs Data Quantity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax = axes[1]
    final_losses = [r["final_loss"] if r["final_loss"] is not None else np.nan for r in results_sparsity]
    ax.plot(n_traj_values, final_losses, marker='s', linewidth=2, color='coral', alpha=0.7)
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('Final Loss')
    ax.set_title('Training Loss vs Data Quantity')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.suptitle('Robustness to Data Sparsity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_ablation_results(hutchinson_results, lambda_results, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    M_values = [r["M"] for r in hutchinson_results]
    param_names = list(hutchinson_results[0]["results"].keys())
    for param_name in param_names:
        errors = [r["results"][param_name]["relative_error"]
                  * 100 for r in hutchinson_results]
        ax.plot(M_values, errors, marker='o',
                linewidth=2, label=param_name, alpha=0.7)
    ax.set_xlabel('Number of Hutchinson Samples (M)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Effect of Hutchinson Estimator Samples')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    lambda_values = [r["lambda"] for r in lambda_results]
    for param_name in param_names:
        errors = [r["results"][param_name]["relative_error"] * 100 for r in lambda_results]
        ax.plot(lambda_values, errors, marker='s', linewidth=2, label=param_name, alpha=0.7)
    ax.set_xlabel('Physics Loss Weight (λ_physics)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Effect of Physics Loss Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax = axes[1, 0]
    mean_errors_hutch = [np.mean([r["results"][p]["relative_error"] for p in param_names]) * 100 for r in hutchinson_results]
    ax.bar([str(m) for m in M_values], mean_errors_hutch, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Hutchinson Samples (M)')
    ax.set_ylabel('Mean Relative Error (%)')
    ax.set_title('Overall Accuracy vs M')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mean_errors_hutch):
        ax.text(i, v, f'{v:.4f}%', ha='center', va='bottom')
    ax = axes[1, 1]
    final_losses = [r["final_loss"] if r["final_loss"] is not None else np.nan for r in lambda_results]
    ax.plot(lambda_values, final_losses, marker='D', linewidth=2, color='coral', alpha=0.7)
    ax.set_xlabel('Physics Loss Weight (λ_physics)')
    ax.set_ylabel('Final Total Loss')
    ax.set_title('Training Loss vs λ_physics')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.suptitle('Ablation Studies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
