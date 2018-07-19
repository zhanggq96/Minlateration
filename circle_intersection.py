import math
import numpy as np


def get_circle_intersections(circle0, circle1):
    # https://stackoverflow.com/questions/3349125/circle-circle-intersection-points
    c0, r0, _ = circle0
    c1, r1, _ = circle1

    # d = math.sqrt((x0-x1)**2 + (y0-y1)**2)
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

    # p2x = x0 + a*(x1 - x0) / d
    # p2y = y0 + a*(y1 - y0) / d
    p2 = c0 + a*(c1 - c0) / d

    # p3x_top = p2x + h*(y1 - y0) / d
    # p3y_top = p2y + h*(x1 - x0) / d
    p3_top = p2 + h*(c1 - c0) / d

    # p3x_bot = p2x + h * (y1 - y0) / d
    # p3y_bot = p2y + h * (x1 - x0) / d
    p3_bot = p2 + h * (c1 - c0) / d

    return p3_top, p3_bot, 'intersect'