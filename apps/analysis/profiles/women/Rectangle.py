class RectangleClassifier:

    @staticmethod
    def classify(measurements):

        shoulder_to_hip = measurements["shoulder_to_hip"]
        waist_to_hip = measurements["waist_to_hip"]

        if (
            0.95 <= shoulder_to_hip <= 1.05 and
            waist_to_hip >= 0.80
        ):
            return {
                "match": True,
                "confidence": 0.84,
                "reasons": [
                    "Straight body proportions",
                    "Minimal waist definition"
                ]
            }

        return {
            "match": False,
            "confidence": 0.0
        }