def compute_scores(features):

    body_balance = features.get("body_balance", 1.0)
    waist_definition = features.get("waist_definition", 0.0)
    visibility = features.get("avg_visibility", 1.0)

    silhouette_balance = int(
        max(
            50,
            min(
                100,
                100 - abs(body_balance - 1.0) * 100
            )
        )
    )

    waist_score = int(
        max(
            40,
            min(
                100,
                waist_definition * 300
            )
        )
    )

    confidence_score = int(visibility * 100)

    return {
        "silhouette_balance": silhouette_balance,
        "waist_definition_score": waist_score,
        "landmark_confidence": confidence_score
    }