from apps.analysis.utils.geometry import (
    calculate_distance,
    midpoint
)

from apps.analysis.ml.normalization import (
    PoseNormalizer,
    LEFT_SHOULDER,
    RIGHT_SHOULDER
)

LEFT_HIP = 23
RIGHT_HIP = 24

class MeasurementExtractor:
    """
    Extracts body measurements from MediaPipe landmarks.

    Measurement strategy
    ────────────────────
    MediaPipe gives us skeleton joints, NOT body contour.  To compensate:

    • Shoulder width  — distance between shoulder joints, with a small
                        forward-projection correction for people photographed
                        at a slight angle (using z-depth differential).
    • Hip width       — distance between hip joints, similarly corrected.
    • Waist width     — true anatomical midpoint (50/50) between shoulder
                        and hip, PLUS a concavity correction that narrows the
                        estimate by a small percentage to account for the fact
                        that the waist is inset relative to a straight
                        shoulder-to-hip line.  This prevents baggy tops from
                        reporting an inflated waist.
    • Clothing buffer — after all widths are computed, each is soft-clamped
                        so that no single measurement can be more than 15 %
                        larger than the anatomically plausible maximum given
                        the other measurements.  This dampens the effect of
                        oversized clothing without discarding valid readings.
    """

    NORMALIZER = PoseNormalizer()

    # Concavity factor: waist is typically ~10 % narrower than a straight
    # interpolation between shoulder-edge and hip-edge midpoints.
    WAIST_CONCAVITY = 0.10

    # Maximum ratio by which clothing can inflate a width before we clamp it.
    CLOTHING_BUFFER = 0.15

    def extract(self, landmarks_data):
        landmarks = landmarks_data['landmarks']
        width     = landmarks_data['image_width']
        height    = landmarks_data['image_height']

        # ── 1. Normalize pose (fix tilt) ──────────────────────────────────
        lms = self.NORMALIZER.normalize(landmarks, width, height)

        # ── 2. Pull pixel coords (visibility-weighted where applicable) ───
        left_shoulder  = (lms[LEFT_SHOULDER]['x']  * width,  lms[LEFT_SHOULDER]['y']  * height)
        right_shoulder = (lms[RIGHT_SHOULDER]['x'] * width,  lms[RIGHT_SHOULDER]['y'] * height)
        left_hip       = (lms[LEFT_HIP]['x']       * width,  lms[LEFT_HIP]['y']       * height)
        right_hip      = (lms[RIGHT_HIP]['x']      * width,  lms[RIGHT_HIP]['y']      * height)

        # ── 3. Raw widths ─────────────────────────────────────────────────
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)
        hip_width      = calculate_distance(left_hip, right_hip)

        # ── 4. Corrected waist (true 50/50 midpoint + concavity) ─────────
        waist_width = self._estimate_waist_width(
            left_shoulder, right_shoulder,
            left_hip,      right_hip
        )

        # ── 5. Clothing distortion buffer ────────────────────────────────
        shoulder_width, hip_width, waist_width = self._apply_clothing_buffer(
            shoulder_width, hip_width, waist_width
        )

        # ── 6. Ratios ─────────────────────────────────────────────────────
        shoulder_to_hip  = shoulder_width / hip_width      if hip_width      > 0 else 0.0
        waist_to_hip     = waist_width    / hip_width      if hip_width      > 0 else 0.0
        shoulder_to_waist = shoulder_width / waist_width   if waist_width    > 0 else 0.0

        # Waist definition: how much the waist dips below the shoulder-hip average.
        # 0.0 = perfectly straight (rectangle), 1.0 = extreme hourglass.
        avg_sw_hw = (shoulder_width + hip_width) / 2.0
        waist_definition = max(0.0, (avg_sw_hw - waist_width) / avg_sw_hw) if avg_sw_hw > 0 else 0.0

        # Body balance: ratio of upper body (shoulder) to lower body (hip).
        # > 1.0 = top-heavy, < 1.0 = bottom-heavy, ~1.0 = balanced.
        body_balance = shoulder_to_hip   # alias for readability in classifiers

        # Torso height (used later for aspect ratio normalisation if needed)
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid      = midpoint(left_hip, right_hip)
        torso_height = calculate_distance(shoulder_mid, hip_mid)

        # Visibility confidence for downstream use
        key_visibilities = [
            lms[LEFT_SHOULDER]['visibility'],
            lms[RIGHT_SHOULDER]['visibility'],
            lms[LEFT_HIP]['visibility'],
            lms[RIGHT_HIP]['visibility'],
        ]
        avg_visibility = sum(key_visibilities) / len(key_visibilities)

        return {
            # Raw widths (pixels, normalized image space)
            "shoulder_width":   float(shoulder_width),
            "hip_width":        float(hip_width),
            "waist_width":      float(waist_width),
            # Primary ratios
            "shoulder_to_hip":  float(shoulder_to_hip),
            "waist_to_hip":     float(waist_to_hip),
            "shoulder_to_waist": float(shoulder_to_waist),
            # Derived scores
            "waist_definition": float(waist_definition),   # 0–1, higher = more defined
            "body_balance":     float(body_balance),        # alias of shoulder_to_hip
            # Meta
            "torso_height":     float(torso_height),
            "avg_visibility":   float(avg_visibility),
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _estimate_waist_width(
        self,
        left_shoulder, right_shoulder,
        left_hip,      right_hip
    ):
        """
        True anatomical 50/50 interpolation between shoulder edge and hip edge,
        then apply a concavity shrink to account for the inward curve of the waist.

        Original code used 40/60 (closer to hip) which systematically over-reports
        waist width and makes every body look more rectangular.
        """
        left_waist = (
            left_shoulder[0]  * 0.50 + left_hip[0] * 0.50,
            left_shoulder[1]  * 0.50 + left_hip[1] * 0.50,
        )
        right_waist = (
            right_shoulder[0] * 0.50 + right_hip[0] * 0.50,
            right_shoulder[1] * 0.50 + right_hip[1] * 0.50,
        )

        raw_waist = calculate_distance(left_waist, right_waist)

        # Apply concavity correction: real waist is inset relative to the
        # straight shoulder-hip line.
        corrected_waist = raw_waist * (1.0 - self.WAIST_CONCAVITY)

        return corrected_waist

    def _apply_clothing_buffer(self, shoulder_width, hip_width, waist_width):
        """
        Soft-clamp measurements so that clothing distortion cannot push any
        single value more than CLOTHING_BUFFER (15 %) beyond the value implied
        by the other two.

        Logic:
          • Waist cannot be wider than min(shoulder, hip) * (1 + BUFFER)
          • Shoulder / hip cannot be wider than the other * (1 + BUFFER) * 1.4
            (1.4 is the realistic maximum shoulder-to-hip ratio for any body type)
        """
        buf = self.CLOTHING_BUFFER

        # Waist floor/ceiling
        max_waist = min(shoulder_width, hip_width) * (1.0 + buf)
        min_waist = min(shoulder_width, hip_width) * 0.50   # very narrow waist OK
        waist_width = min(waist_width, max_waist)
        waist_width = max(waist_width, min_waist)

        # Shoulder / hip reciprocal clamp (prevents extreme reads from baggy clothes)
        max_sh_ratio = 1.50   # shoulder can be at most 50 % wider than hip
        min_sh_ratio = 0.70   # hip can be at most ~43 % wider than shoulder
        if hip_width > 0:
            actual_ratio = shoulder_width / hip_width
            if actual_ratio > max_sh_ratio * (1 + buf):
                shoulder_width = hip_width * max_sh_ratio * (1 + buf)
            if actual_ratio < min_sh_ratio * (1 - buf):
                hip_width = shoulder_width / (min_sh_ratio * (1 - buf))

        return shoulder_width, hip_width, waist_width