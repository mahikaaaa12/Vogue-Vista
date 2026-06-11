class OvalClassifier:

    @staticmethod
    def classify(measurements):

        waist_to_hip = measurements["waist_to_hip"]

        if waist_to_hip > 0.95:

            return {
                "match": True,
                "confidence": 0.85,
                "reasons": [
                    "Rounded midsection",
                    "Less torso taper"
                ]
            }

        return {
            "match": False,
            "confidence": 0.0
        }