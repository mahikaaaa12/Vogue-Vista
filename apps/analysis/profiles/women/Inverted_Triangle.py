class InvertedTriangleClassifier:

    @staticmethod
    def classify(measurements):

        shoulder_to_hip = measurements["shoulder_to_hip"]

        if shoulder_to_hip > 1.08:

            return {
                "match": True,
                "confidence": 0.89,
                "reasons": [
                    "Broader shoulders",
                    "Narrower hips"
                ]
            }

        return {
            "match": False,
            "confidence": 0.0
        }