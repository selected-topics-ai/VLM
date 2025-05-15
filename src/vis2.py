from matplotlib import pyplot as plt

def make_domain_plots(domain_metrics):

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    axs = axs.flatten()

    for ax, domain in zip(axs, ["in-domain", "near-domain", "out-domain"]):
        ax.hist(domain_metrics[domain]['CIDEr-BLIP']['scores_per_sample'], bins=100, alpha=0.5, label="BLIP")
        ax.hist(domain_metrics[domain]['CIDEr-LLaVA']['scores_per_sample'], bins=100, alpha=0.5, label="LLaVA")
        ax.set_title(f"{domain}")
        ax.grid(visible=True, alpha=0.3)
        ax.set_xlabel(f"CIDEr Score")
        ax.set_ylabel("Image Count")

        if domain == "in-domain":
            y_max = 20
        if domain == "near-domain":
            y_max = 150
        if domain == "out-domain":
            y_max = 120

        ax.vlines(x=0.5, ymin=0, ymax=y_max, color="r", label=f"CIDEr goodness threshold")

    plt.legend()
    plt.show()