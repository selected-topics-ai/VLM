import numpy as np

def count_images_by_treshold(metrics: dict, treshold: float=0.5) -> dict:

    threshold = 0.5

    overall_by_threshold = {
        "BLIP": {
            "in-domain": {
                "above": 0,
                "below": 0,
            },
            "out-domain": {
                "above": 0,
                "below": 0,
            },
            "near-domain": {
                "above": 0,
                "below": 0,
            },
            "overall": {
                "above": 0,
                "below": 0,
            },
        },
        "LLaVA": {
            "in-domain": {
                "above": 0,
                "below": 0,
            },
            "out-domain": {
                "above": 0,
                "below": 0,
            },
            "near-domain": {
                "above": 0,
                "below": 0,
            },
            "overall": {
                "above": 0,
                "below": 0,
            },
        }
    }
    for model in ["BLIP", "LLaVA"]:
        for domain in ["in-domain", "near-domain", "out-domain"]:
            scores_per_sample = metrics[domain][f"CIDEr-{model}"]["scores_per_sample"]
            overall_by_threshold[model][domain]["above"] += np.sum(scores_per_sample >= threshold)
            overall_by_threshold[model][domain]["below"] += np.sum(scores_per_sample < threshold)
            overall_by_threshold[model]["overall"]["above"] += np.sum(scores_per_sample >= threshold)
            overall_by_threshold[model]["overall"]["below"] += np.sum(scores_per_sample < threshold)

    return overall_by_threshold