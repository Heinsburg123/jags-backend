import pangolin as pg
from pangolin.ir import RV, Add, Constant, Normal
from include import Sample_prob


a = RV(Constant(3.0))
b = RV(Constant(4.0))
c = RV(Constant(1.0))
mu = RV(Add(), a, b)
y = RV(Normal(), mu, c)
z = RV(Add(), y, mu)
sp = Sample_prob()

print(sp.sample([mu, z], {}))




