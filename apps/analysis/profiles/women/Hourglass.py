class HourglassClassifier:

    @staticmethod
    def classify(measurements):

        shoulder_to_hip = measurements["shoulder_to_hip"]
        waist_to_hip = measurements["waist_to_hip"]

        if (
            0.95 <= shoulder_to_hip <= 1.05 and
            waist_to_hip < 0.75
        ):
            return {
                "match": True,
                "confidence": 0.92,
                "reasons": [
                    "Balanced shoulders and hips",
                    "Defined waistline"
                ]
            }

        return {
            "match": False,
            "confidence": 0.0
        }