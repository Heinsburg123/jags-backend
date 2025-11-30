import pangolin as pg
from pangolin.ir import *
from include import Sample_prob

loc = RV(Constant(2))
scale = RV(Constant(0.5))

# Autoregressive Normal of length 6
ar_normal = Autoregressive(
    base_op=Normal(),
    length=6,
    in_axes=[None],  # loc and scale constant per step
    where_self=0
)

# Map the whole 6-step AR process across axis 0 with size 4
vmapped_ar = RV(
    VMap(ar_normal, in_axes=(None, None), axis_size=4),
    loc, scale
)


sp = Sample_prob()
print(sp.sample([vmapped_ar], {}))



