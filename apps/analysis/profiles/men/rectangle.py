class RectangleClassifier:

    @staticmethod
    def classify(measurements):

        shoulder_to_hip = measurements["shoulder_to_hip"]
        waist_to_hip = measurements["waist_to_hip"]

        if (
            0.95 <= shoulder_to_hip <= 1.05 and
            0.85 <= waist_to_hip <= 1.0
        ):

            return {
                "match": True,
                "confidence": 0.82,
                "reasons": [
                    "Straight torso structure",
                    "Even proportions"
                ]
            }

        return {
            "match": False,
            "confidence": 0.0
        }