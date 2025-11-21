import pangolin as pg
from pangolin.ir import *
from include import Sample_prob

op = SimpleIndex()
x = RV(Constant([[2,3,1], [4,1,5], [7,2,6]]))
z = RV(Sum(1), x)
sp = Sample_prob()
print(sp.sample([z], {}))



