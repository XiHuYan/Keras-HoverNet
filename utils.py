import skimage
import skimage.measure
import numpy as np


def norm(x, a, b):
    t = a + (b-a)*(x-x.min())/(x.max()-x.min())
    return t

def get_hv_dist(lbl):   
    lbl = np.digitize(lbl, np.unique(lbl))-1
    regs = skimage.measure.regionprops(lbl)
    results = np.zeros(lbl.shape+(2,))
    for reg in regs:
        cent = reg.centroid
        coords = reg.coords
        h_dist = norm(coords[:,1]-cent[1], -1, 1)
        v_dist = norm(coords[:,0]-cent[0], -1, 1)
        results[coords[:,0],coords[:,1], 0] = h_dist
        results[coords[:,0],coords[:,1], 1] = v_dist
    return results

