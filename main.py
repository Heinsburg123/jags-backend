import pangolin as pg
from pangolin.ir import *
from include import Sample_prob

x = RV(Constant([[1,2,3], [4,5,6]]))
c = RV(Constant([0, 1]))
d = RV(Constant([.1, .2, .7]))
id2 = RV(Categorical(), d)
z = RV(SimpleIndex(), x, c, id2)

sp = Sample_prob()
print(sp.sample([z], {}))



