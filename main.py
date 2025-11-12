import pangolin as pg
from pangolin.ir import *
from include import Sample_prob

a = RV(Constant([[1,2,3], [2,3,4]]))
b = RV(Constant([[1,2], [2,3], [3,4]]))
c = RV(MatMul(), a, b)
sp = Sample_prob() 
print(sp.sample([c], {}))




