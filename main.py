import pangolin as pg
from pangolin.ir import *
from include import Sample_prob
import numpy as np


sp = Sample_prob()

a = RV(Constant([[[1,2,3],[2,3,4]], [[1,2,3], [2,3,4]]]))
b = RV(Sum(axis=0), a)
print(sp.sample([b], [], []))

