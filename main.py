import pangolin as pg
from pangolin.ir import RV, Add, Constant, Normal
from include import Sample_prob


x = RV(Constant(3))
y = RV(Constant(4))
z = RV(Normal(), x, y)
k = RV(Add(), z, x)

sp = Sample_prob()

print(sp.sample(z, {k:7}))




