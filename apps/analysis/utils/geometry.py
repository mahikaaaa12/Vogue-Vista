import math

def calculate_distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )

def midpoint(a, b):
    return (
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0
    )

def rotate_point(px, py, cx, cy, angle_rad):

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    dx = px - cx
    dy = py - cy

    return (
        cx + dx * cos_a - dy * sin_a,
        cy + dx * sin_a + dy * cos_a
    )