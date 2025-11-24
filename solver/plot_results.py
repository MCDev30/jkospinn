from pkg import *
from config import *


def plot_results(history, results, sde, save_path=None):
    import math

    # ----------- SCIENTIFIC SYMBOL MAP -----------
    param_latex = {
        "sigma": r"$\Sigma$", 
        "mu": r"$\mu$",
        "alpha": r"$\alpha$",
        "D": r"$D$",
        "beta": r"$\beta$",
        "theta": r"$\theta$",
        "gamma": r"$\gamma$",
    }

    def get_param_symbol(key):
        return param_latex.get(key, key)

    # === Prepare epochs vector
    epochs = np.arange(len(history["loss_total"])) * 5

    # === Prepare beautiful scientific-style plot aesthetics
    plt.style.use("seaborn-v0_8-dark-palette")
    plt.rcParams.update({
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1,
        "grid.color": "#cccccc",
        "font.family": "serif",
        'mathtext.fontset': 'stix',
        'font.serif': ['Times New Roman'],
    })

    # === Prepare subplot grid: 2 rows x 3 columns (total 6 subplots)
    # ADJUST FIGURE SIZE FOR LESS FLATTENED PLOTS
    fig, axes = plt.subplots(2, 3, figsize=(
        18, 8), constrained_layout=False)  

    # --- (0, 0): Loss curves ---
    ax_loss = axes[0, 0]
    loss_total = np.array(sorted(history["loss_total"], reverse=True))
    loss_data = np.array(sorted(history["loss_data"], reverse=True))
    loss_physics = np.array(sorted(history["loss_physics"], reverse=True))
    # -- Correction HERE: Use correct mathcal -- #
    ax_loss.plot(epochs, loss_total,
                 label=r"$\mathcal{L}_{\mathrm{tot}}$", alpha=0.8, linewidth=2, color="#0057b8")
    ax_loss.plot(epochs, loss_data, label=r"$\mathcal{L}_{\mathrm{data}}$",
                 alpha=0.8, linestyle='-.', linewidth=2, color="#ff5a36")
    ax_loss.plot(epochs, loss_physics, label=r"$\mathcal{L}_{\mathrm{phy}}$",
                 alpha=0.8, linestyle=':', linewidth=2, color="#3ab795")
    ax_loss.set_xlabel("Epoch", labelpad=8)
    ax_loss.set_ylabel("Loss", labelpad=8)
    ax_loss.set_yscale("log")
    ax_loss.legend(frameon=True, facecolor='white', edgecolor='#303030')
    ax_loss.set_title("Training Loss", fontsize=18, pad=9)
    ax_loss.grid(True, alpha=0.18, linestyle='--', linewidth=0.8)

    # --- (0, 1) and (0,2) and (1,0)/(1,1)/...: Parameter convergence, show up to 5 parameters
    true_params = sde.get_true_params()
    param_keys = list(history["params_phys"].keys())
    max_param_plots = 5
    plotted = 0

    for idx, key in enumerate(param_keys[:max_param_plots]):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        ax = axes[row, col]

        # Get proper label
        param_label = get_param_symbol(key)
        # Also if D, we represent as Sigma (sum style)!
        if key == "D":
            sigma_hist = np.sqrt(np.array(history["params_phys"]["D"]))
            # Use grand sigma for display (matches param_latex above)
            ax.plot(epochs, sigma_hist,
                    label=r"$\Sigma$ (est)", alpha=0.9, linewidth=2.2, color="#b85c00")
            # On the true value, use the same grand Sigma symbol
            ax.axhline(true_params["sigma"], linestyle="--", color="#2a2d34",
                       linewidth=1.8, label=r"$\Sigma$ (true)")
            ylabel = r"$\Sigma$"
            # Use raw string for title to avoid invalid escape warning
            ax.set_title(r"Convergence: $\Sigma$", fontsize=16)
        else:
            col_color = "#0081a7" if key in [
                'mu', 'alpha', 'sigma', 'beta', 'gamma'] else "#de425b"
            ax.plot(
                epochs,
                history["params_phys"][key],
                label=f"{param_label} (est)",
                alpha=0.93,
                linewidth=2.4,
                color=col_color
            )
            ax.axhline(
                true_params[key],
                linestyle="--",
                color="#262626",
                linewidth=1.8,
                label=f"{param_label} (true)"
            )
            ylabel = f"{param_label}"
            ax.set_title(f"Convergence: {param_label}", fontsize=16)

        ax.set_xlabel("Epoch", labelpad=7)
        ax.set_ylabel(ylabel, labelpad=6)
        ax.legend(loc="best", frameon=True, fancybox=True,
                  facecolor='white', edgecolor="#333")
        ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.9)
        plotted += 1

    # --- Accuracy barplot (1, 2)
    ax_err = axes[1, 2]
    errors = [results[k]["relative_error"] * 100 for k in results.keys()]
    param_names = [get_param_symbol(k) for k in results.keys()]
    try:
        colors = sns.color_palette("rocket", len(param_names))
    except Exception:
        colors = None
    bars = ax_err.bar(param_names, errors, color=colors,
                      alpha=0.95, linewidth=2, edgecolor="k")
    ax_err.set_ylabel("Relative Error (%)", labelpad=7)
    ax_err.set_title("Parameter Estimation Accuracy", fontsize=16, pad=8)
    ax_err.grid(True, alpha=0.18, axis="y", linestyle='--', linewidth=0.8)

    # --- Place bar labels closer to the bars
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        # Use a small offset, proportional if necessary, to keep texts close to the top
        # If errors are small, don't add much
        offset = 0.001 * max(1.0, height)
        ax_err.text(
            bar.get_x() + bar.get_width()/2., height + offset,
            '{:.4f}%'.format(err),
            ha='center', va='bottom',
            fontsize=11, fontweight='bold', color="#2a2a2a"
        )

    # --- Table of results (1, 1) if unoccupied, else (1, 0)
    table_slot_used = [False] * 3
    for idx in range(3):
        if plotted >= idx + 3:
            table_slot_used[idx] = True
    table_row = 1
    table_col = 1 if not table_slot_used[1] else (
        0 if not table_slot_used[0] else 2)
    ax_tbl = axes[table_row, table_col]
    ax_tbl.axis('tight')
    ax_tbl.axis('off')
    table_data = []
    for key in results.keys():
        symbol = get_param_symbol(key)
        # Special case: for D display result in 'Sigma' format
        if key == "D":
            # We print label as grand Sigma, and estimated/true as sigma params (the true/est value shown is sigma not D)
            table_data.append([
                r"$\Sigma$",
                "{:.2f}".format(results[key]['true']),
                "{:.8f}".format(results[key]['estimated']),
                "{:.4f}%".format(results[key]['relative_error']*100)
            ])
        else:
            table_data.append([
                symbol,
                "{:.2f}".format(results[key]['true']),
                "{:.8f}".format(results[key]['estimated']),
                "{:.4f}%".format(results[key]['relative_error']*100)
            ])
    col_labels = [
        "Parameter",
        "True",
        "Estimated",
        "RE (%)"
    ]
    table = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.13, 2.1)
    for i in range(4):
        table[(0, i)].set_facecolor('#303964')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for j in range(len(table_data)):
        for i in range(4):
            table[(j+1, i)].set_text_props(fontsize=13)

    # --- Hide unused axes (if any)
    for r in range(2):
        for c in range(3):
            ax = axes[r, c]
            # Skip those where we've already drawn
            if (r, c) == (0, 0):
                continue  # Loss
            if (r == 1 and c == 2):
                continue  # Barplot
            if (r == table_row and c == table_col):
                continue  # Table
            # parameter plots offset -1 because (0,0) is losses
            idx_flat = r * 3 + c - 1
            if idx_flat < plotted:
                continue
            ax.axis('off')

    fig.suptitle(f"JKO-SPINN: {sde.name} Process",
                 fontsize=21, fontweight='bold', color="#000000")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90,
                        bottom=0.08, wspace=0.21, hspace=0.28)
    if save_path:
        fig.savefig(save_path.replace(".png", "_all.png"), dpi=250, bbox_inches='tight', transparent=True)
    plt.show()
