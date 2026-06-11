class TrapezoidClassifier:

    @staticmethod
    def classify(measurements):

        shoulder_width = measurements["shoulder_width"]
        waist_width = measurements["waist_width"]

        ratio = shoulder_width / waist_width

        if 1.2 <= ratio <= 1.5:

            return {
                "match": True,
                "confidence": 0.91,
                "reasons": [
                    "Broad shoulders",
                    "Defined upper body taper"
                ]
            }

        return {
            "match": False,
            "confidence": 0.0
        }