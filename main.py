import pangolin as pg
from pangolin.ir import *
from jags_pangolin.src.engine import Sample_prob
import numpy as np


sp = Sample_prob()

a = RV(Constant([[1,2,3],[2,3,4], [3,3,3]]))
b = RV(Constant([1,2,3]))
c = RV(Matmul(), a, b)
[samp] = sp.sample([c], [], [])

