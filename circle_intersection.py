import math
import numpy as np


def get_circle_intersections(circle0, circle1):
    # https://stackoverflow.com/questions/3349125/circle-circle-intersection-points
    c0, r0, _, _ = circle0
    c1, r1, _, _ = circle1

    d = np.linalg.norm(c0-c1)

    if d > (r0 + r1):
        return None, None, 'seperate'
    elif d < abs(r0 - r1):
        # return (is circle 0 bigger?, is circle 1 bigger?, contained.)
        return r0 >= r1, r1 > r0, 'contained'
    elif d == 0 and r0 == r1:
        return None, None, 'coincident'

    a = (r0**2 - r1**2 + d**2) / (2*d)
    h = math.sqrt(r0**2 - a**2)

    # middle
    p2 = c0 + a*(c1 - c0) / d

    # intersections
    p3_top, p3_bot = np.zeros_like(p2), np.zeros_like(p2)

    # x coord
    p3_top[0] = p2[0] + h*(c1[1] - c0[1]) / d
    # y coord
    p3_top[1] = p2[1] - h*(c1[0] - c0[0]) / d

    # x coord
    p3_bot[0] = p2[0] - h*(c1[1] - c0[1]) / d
    # y coord
    p3_bot[1] = p2[1] + h*(c1[0] - c0[0]) / d

    return p3_top, p3_bot, 'intersect'