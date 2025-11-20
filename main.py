import pangolin as pg
from pangolin.ir import *
from include import Sample_prob

op = SimpleIndex()
x = RV(Constant([[[0.2, 0.5, 0.3], [0.1, 0.6, 0.3]], [[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]]]))
a = RV(Constant([0, 1]))
b = RV(Constant([1]))
c = RV(Constant([0, 1]))
y = RV(op, x, a, b, c)
sp = Sample_prob()
print(sp.sample([y], {}))



