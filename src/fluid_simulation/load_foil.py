import sys
import numpy as np
import re  # regexp
import scipy.interpolate as i
import numpy as np
import matplotlib.pyplot as plt


################################################################
# Airfoil : load profile of a wing
#
# Reads a file whose lines contain coordinates of points,
# separated by an empty line.
# Every line not containing a couple of floats is discarded.
# Returns a couple constitued of the list of points of the
# extrados and the intrados.
def load_foil(file):
    f = open(file, 'r')
    def matchline(line): return re.match(r"\s*([\d\.-]+)\s*([\d\.-]+)", line)
    extra = []
    intra = []
    rextra = False
    rintra = False
    rheader = False
    for line in f:
        m = matchline(line)
        if m is None:
            if not rheader:
                rheader = True
            elif not rextra:
                rextra = True
            elif not rintra:
                rintra = True
            continue
        if (m is not None) and rheader and not rextra and not rintra:
            dim = np.array(list(map(lambda t: float(t), m.groups())))
            continue
        if (m is not None) and rheader and rextra and not rintra:
            extra.append(m.groups())
            continue
        if (m is not None) and rheader and rextra and rintra:
            intra.append(m.groups())
            continue
    ex = np.array(list(map(lambda t: float(t[0]), extra)))
    ey = np.array(list(map(lambda t: float(t[1]), extra)))
    ix = np.array(list(map(lambda t: float(t[0]), intra)))
    iy = np.array(list(map(lambda t: float(t[1]), intra)))
    return dim, ex, ey, ix, iy


def interpolate_matrix(path, start_idx=20, end_idx=80, height=1):
    dim, ex, ey, ix, iy = load_foil(path)
    xs = np.linspace(0, 1, end_idx - start_idx + 1)

    cs_e = i.CubicSpline(ex, ey, axis=0, bc_type='not-a-knot')
    cs_i = i.CubicSpline(ix, iy, axis=0, bc_type='not-a-knot')

    interpolated_matrix = np.zeros((2, end_idx - start_idx + 1))

    interpolated_matrix[0, :] = cs_e(xs)
    interpolated_matrix[1, :] = cs_i(xs)
    
    low_e, high_e = np.min(interpolated_matrix[0, :]), np.max(interpolated_matrix[0, :])
    low_i, high_i = np.min(interpolated_matrix[1, :]), np.max(interpolated_matrix[1, :])
    
    interpolated_matrix[0, :] = (interpolated_matrix[0, :] * height).astype(int)
    interpolated_matrix[1, :] = (interpolated_matrix[1, :] * height).astype(int)

    return xs, interpolated_matrix

if __name__ == "__main__":
    (dim, ex, ey, ix, iy) = load_foil("data/geo05k.dat")
    print(dim)
    print(ex)
    print(ey)
    print(ix)
    print(iy)
    
    xs, M = interpolate_matrix("data/geo05k.dat", 10, 90, 1000)
    
    plt.plot(xs, M[0, :], label='extrados')
    plt.plot(xs, M[1, :], label='intrados')
    plt.legend()
    plt.show()