import pangolin as pg
from pangolin.ir import *
from include import Sample_prob

op = Composite(2, [Add(), Mul()], [[0, 0], [2, 1]])

x = RV(Constant(3.3))
y = RV(Constant(4.4))
z = RV(op, x, y)
sp = Sample_prob()
print(sp.sample([z], {}))




