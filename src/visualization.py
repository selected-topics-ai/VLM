from matplotlib import pyplot as plt

def make_overall_plots(domain_metrics):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs = axs.flatten()

    i = 0

    for ax, model in zip(axs, ["BLIP", "LLaVA"]):
        metric = f"CIDEr-{model}"

        ax.hist(domain_metrics['near-domain'][metric]['scores_per_sample'], bins=100, alpha=0.5,
                label="in-domain" if i == 0 else None)
        ax.hist(domain_metrics['out-domain'][metric]['scores_per_sample'], bins=100, alpha=0.5,
                label="out-domain" if i == 0 else None)
        ax.hist(domain_metrics['in-domain'][metric]['scores_per_sample'], bins=100, alpha=0.5,
                label="in-domain" if i == 0 else None)
        ax.set_title(model)
        ax.grid(visible=True, alpha=0.3)
        ax.set_xlabel("CIDEr Score")
        ax.set_ylabel("Image Count")
        ax.vlines(x=0.5, ymin=0, ymax=150, color="r", label=f"CIDEr goodness threshold" if i == 0 else None)

        i += 1

    fig.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.tight_layout()
    fig.show()

