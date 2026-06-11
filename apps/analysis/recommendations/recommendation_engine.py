from apps.analysis.profiles.female_profiles import FEMALE_BODY_PROFILES
from apps.analysis.profiles.male_profiles import MALE_BODY_PROFILES


def generate_recommendations(gender, shape, features):

    gender = (gender or "female").lower()

    shape = (
        (shape or "")
        .lower()
        .replace(" ", "_")
    )

    print("NORMALIZED SHAPE:", shape)

    if gender == "male":
        profile = MALE_BODY_PROFILES.get(shape, {})
    else:
        profile = FEMALE_BODY_PROFILES.get(shape, {})

    print("PROFILE FOUND:", profile)

    recommendations = {
        "tops": profile.get("tops", []),
        "bottoms": profile.get("bottoms", []),
        "dresses": profile.get("dresses", []),
        "ethnicwear": profile.get("ethnicwear", []),
        "officewear": profile.get("officewear", []),
        "casualwear": profile.get("casualwear", []),
        "footwear": profile.get("footwear", []),
        "accessories": profile.get("accessories", []),
        "avoid": profile.get("avoid", []),
        "styling_tips": profile.get("styling_tips", [])
    }

    body_balance = features.get("body_balance", 1.0)
    waist_definition = features.get("waist_definition", 0.0)

    adaptive_tips = []

    if body_balance > 1.1:
        adaptive_tips.append(
            "Balance broader upper proportions with fuller lower silhouettes."
        )

    elif body_balance < 0.9:
        adaptive_tips.append(
            "Add visual structure to the upper body for balance."
        )

    if waist_definition > 0.22:
        adaptive_tips.append(
            "Structured waist emphasis complements your proportions."
        )

    if waist_definition < 0.12:
        adaptive_tips.append(
            "Layering and contour styling can help create more waist definition."
        )

    recommendations["adaptive_tips"] = adaptive_tips

    return recommendations