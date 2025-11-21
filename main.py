import pangolin as pg
from pangolin.ir import *
from include import Sample_prob

a = RV(Constant(value=[[1,2,3],[4,5,6]]))
b = RV(Sum(1), a)
c = RV(Sum(0), b)



sp = Sample_prob()
print(sp.sample([c], {}))



