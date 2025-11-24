import pangolin as pg
from pangolin.ir import *
from include import Sample_prob

a = RV(Constant(3))
b = RV(Constant(2))
c = RV(Gamma(), a,b)
sp = Sample_prob()
print(sp.sample([c], {}))



