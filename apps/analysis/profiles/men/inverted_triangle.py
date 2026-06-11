class InvertedTriangleClassifier:

    @staticmethod
    def classify(measurements):

        shoulder_to_hip = measurements["shoulder_to_hip"]

        if shoulder_to_hip > 1.12:

            return {
                "match": True,
                "confidence": 0.90,
                "reasons": [
                    "Strong upper body",
                    "Narrow waist and hips"
                ]
            }

        return {
            "match": False,
            "confidence": 0.0
        }