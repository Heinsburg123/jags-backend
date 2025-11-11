import pangolin as pg
from pangolin.ir import *
from include import Sample_prob


a = RV(Constant(2))
b = RV(Constant(5))
c = RV(Constant(3))
d = RV(BetaBinomial(), a,b,c)

sp = Sample_prob()

print(sp.sample([d], {}))




