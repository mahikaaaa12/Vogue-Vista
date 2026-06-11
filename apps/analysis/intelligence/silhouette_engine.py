def silhouette_balance(features):

    symmetry = features.get("symmetry")

    if symmetry > 0.92:
        return "Highly balanced silhouette"

    return "Moderately balanced silhouette"