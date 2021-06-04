

import numpy as np

def l2(x1, x2):
    distance = np.mean((x1-x2)**2)

    return distance
