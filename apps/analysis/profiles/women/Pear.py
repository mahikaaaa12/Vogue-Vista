class PearClassifier:

    @staticmethod
    def classify(measurements):

        shoulder_to_hip = measurements["shoulder_to_hip"]
        waist_to_hip = measurements["waist_to_hip"]

        if (
            shoulder_to_hip < 0.95 and
            waist_to_hip < 0.85
        ):
            return {
                "match": True,
                "confidence": 0.88,
                "reasons": [
                    "Hips wider than shoulders",
                    "Lower body dominance"
                ]
            }

        return {
            "match": False,
            "confidence": 0.0
        }