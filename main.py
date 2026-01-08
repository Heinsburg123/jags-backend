from jags_pangolin import run_model
from pangolin.ir import * 

a = RV(Constant([[1,2,3],[2,3,4], [3,3,3]])) 
b = RV(Constant([1,2,3])) 
c = RV(Matmul(), a, b) 
[samp] = run_model([c], [], [], ninter=10)
print(samp)