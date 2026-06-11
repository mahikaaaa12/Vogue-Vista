class AppleClassifier:

    @staticmethod
    def classify(measurements):

        waist_to_hip = measurements["waist_to_hip"]

        if waist_to_hip >= 0.90:

            return {
                "match": True,
                "confidence": 0.86,
                "reasons": [
                    "Wider midsection",
                    "Less waist definition"
                ]
            }

        return {
            "match": False,
            "confidence": 0.0
        }