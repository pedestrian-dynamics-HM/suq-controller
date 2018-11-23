#!/usr/bin/env python3 

# TODO: """ << INCLUDE DOCSTRING (one-line or multi-line) >> """

import numpy as np

from suqc.two_density_dmap import DMAPWrapper
from suqc.two_density_data import load_data, FILE_ACCUM

# --------------------------------------------------
# people who contributed code
__authors__ = "Daniel Lehmberg"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


df = load_data(FILE_ACCUM)

data = df.values.T

sdata = data[:, :-1]
edata = data[:, 1:]

K = edata @ np.linalg.pinv(sdata)

evalK, evecK = np.linalg.eig(K)

dt = 1

omega = np.log(evalK) / dt

b = np.linalg.pinv(evecK) @ np.array([30, 0, 0, 0])

NT = 5

res = np.zeros([NT])

for i in range(NT):
    evecK @ np.exp(omega) @ b