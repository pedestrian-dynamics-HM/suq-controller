#!/usr/bin/env python3 

# TODO: """ << INCLUDE DOCSTRING (one-line or multi-line) >> """

import numpy as np

from suqc.two_density_dmap import DMAPWrapper

# --------------------------------------------------
# people who contributed code
__authors__ = "Daniel Lehmberg"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


dmap = DMAPWrapper.load_pickle()

phi = dmap.eigenvectors.T  # TODO: orthogonalize this...

phi_old = phi[:, :-1]
phi_new = phi[1, :]

K = np.linalg.lstsq(phi_new, phi_old, rcond=1E-13)




