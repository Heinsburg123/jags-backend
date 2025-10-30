import pangolin as pg
from pangolin.ir import RV, Add, Constant, Normal
from include import Sample_prob


a = RV(Constant([2,3]))
b = RV(Constant([2,3]))
c = RV(Constant(1.0))
sp = Sample_prob()

print(sp.sample([a], {}))




