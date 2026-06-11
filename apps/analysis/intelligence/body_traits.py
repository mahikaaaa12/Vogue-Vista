def extract_traits(features):

    traits = []

    shoulder_ratio = features.get("shoulder_to_hip", 1.0)
    waist_definition = features.get("waist_definition", 0.0)
    visibility = features.get("avg_visibility", 1.0)

    if shoulder_ratio > 1.08:
        traits.append("Shoulder dominant")

    elif shoulder_ratio < 0.92:
        traits.append("Hip dominant")

    else:
        traits.append("Balanced frame")

    if waist_definition > 0.22:
        traits.append("Defined waist")

    elif waist_definition < 0.12:
        traits.append("Straight waistline")

    else:
        traits.append("Soft waist definition")

    if visibility > 0.9:
        traits.append("High landmark confidence")

    return traits