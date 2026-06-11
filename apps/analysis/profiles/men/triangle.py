class TriangleClassifier:

    @staticmethod
    def classify(measurements):

        shoulder_to_hip = measurements["shoulder_to_hip"]

        if shoulder_to_hip < 0.95:

            return {
                "match": True,
                "confidence": 0.87,
                "reasons": [
                    "Waist and hips wider than shoulders"
                ]
            }

        return {
            "match": False,
            "confidence": 0.0
        }