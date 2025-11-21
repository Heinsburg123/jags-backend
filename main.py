import pangolin as pg
from pangolin.ir import *
from include import Sample_prob

b = RV(Constant([[2, 1], [3, 2]]))
c = RV(Inv(), b)
sp = Sample_prob()
print(sp.sample([c], {}))



