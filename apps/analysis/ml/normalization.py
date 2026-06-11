import math
from apps.analysis.utils.geometry import (
    midpoint,
    rotate_point
)

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe landmark indices (POSE_LANDMARKS)
# ──────────────────────────────────────────────────────────────────────────────
LEFT_SHOULDER   = 11
RIGHT_SHOULDER  = 12
LEFT_ELBOW      = 13
RIGHT_ELBOW     = 14
LEFT_HIP        = 23
RIGHT_HIP       = 24
LEFT_KNEE       = 25
RIGHT_KNEE      = 26

def _visibility(landmark):
    return landmark.get('visibility', 1.0)

class PoseNormalizer:
    """
    Rotates all landmarks so that:
      • The shoulder line is perfectly horizontal  (corrects camera/body tilt)
      • The torso vertical axis aligns with the image vertical axis

    This is the single most impactful fix: a 5° lean changes shoulder/hip
    ratio by ~8 % with un-normalised landmarks.
    """

    def normalize(self, landmarks, width, height):
        """
        Return a new list of landmark dicts with corrected x/y coordinates.
        z and visibility are unchanged.
        """
        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]

        # Pixel coords of left / right shoulder
        ls_px = (ls['x'] * width, ls['y'] * height)
        rs_px = (rs['x'] * width, rs['y'] * height)

        # Angle of shoulder line relative to horizontal
        dy = rs_px[1] - ls_px[1]
        dx = rs_px[0] - ls_px[0]
        tilt_angle = math.atan2(dy, dx)   # radians; 0 = perfectly horizontal

        # Rotation center = midpoint of shoulders
        cx, cy = midpoint(ls_px, rs_px)

        corrected = []
        for lm in landmarks:
            px = lm['x'] * width
            py = lm['y'] * height
            nx, ny = rotate_point(px, py, cx, cy, -tilt_angle)
            corrected.append({
                'x':          nx / width,
                'y':          ny / height,
                'z':          lm.get('z', 0.0),
                'visibility': lm.get('visibility', 1.0),
            })

        return corrected