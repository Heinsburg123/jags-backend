import numpy as np
from pangolin.ir import *
from  Backend.scalar_ops import Scalar_ops

class Multi_funcs:
    def Matmul(n, res:dict):
        parent1 = res[n].parents[0]
        parent2 = res[n].parents[1]
        return f"{n} <- v{parent1._n} %*% v{parent2._n}"

    def Sum(n, res:dict, tmp_res):
        def cover(arr, name):
            code = ""
            def loop(arr, indices, name):
                nonlocal code
                if len(indices) == arr.ndim:
                    code += name + "["
                    for i in range(len(indices)):
                        code += f"{indices[i]+1},"
                    code = code[:-1] + "]"
                    code += f"<- {arr[tuple(indices)]}\n"
                else:
                    for i in range(arr.shape[len(indices)]):
                        loop(arr, indices + [i], name)
            loop(arr, [], name)
            return code
        name = res[n].parents[0].op.name
        if(name == "VMap" or name == "Multinomial" or name == "Dirichlet" or name == "MultiNormal" or (name == "Sum" and res[n].parents[0].ndim==1) or (name == "Inv" and res[n].parents[0].ndim==1)):
            return f"{n} <- sum(v{res[n].parents[0]._n}[])\n"
        if(name == "Constant"):
            parent = res[n].parents[0].op.value
        elif(name == "Sum" or name == "Inv"):
            parent = tmp_res[f"v{res[n].parents[0]._n}"]
        axis = res[n].op.axis
        arr = np.array(parent)
        ans = np.sum(arr, axis)
        code = cover(ans, n)
        tmp_res[n] = ans.tolist()
        return code

    def Inv(n, res:dict, tmp_res):
        name = res[n].parents[0].op.name
        if(name == "Constant"):
            parent = res[n].parents[0].op.value
        elif(name == "Sum" or name == "Inv"):
            parent = tmp_res[f"v{res[n].parents[0]._n}"]
        arr = np.array(parent)
        ans = np.linalg.inv(arr)
        ans = np.round(ans, decimals=6)
        code = ""
        for i in range(ans.shape[0]):
            for j in range(ans.shape[1]):
                code += f"{n}[{i+1},{j+1}] <- {ans[i][j]}\n"
        tmp_res[n] = ans.tolist()
        return code
    
    def Softmax(n, res:dict):
        parent1 = res[n].parents[0]
        k = parent1.shape[0]
        code += f"for (i in 1:{k})" + "{\n"
        code += f"  {n}[i] <- exp(v{parent1._n}[i])/sum(exp(v{parent1._n}[])[])\n" + "}\n"
        return code
    
    def MultiNormal(n, res:dict):
        parent1 = res[n].parents[0]
        parent2 = res[n].parents[1]
        p = res[n].shape[0]
        return f"{n}[1:{p}] ~ dmnorm(v{parent1._n}[1:{p}], v{parent2._n}[1:{p},1:{p}])"
    
    def MultiNominal(n, res:dict):
        parent1 = res[n].parents[0]
        n = res[n].shape[0]
        p = res[n].shape[1]
        return f"{n}[1:{p}] ~ dmulti(v{parent1._n}[1:{p}], {n})"

    def Dirichlet(n, res:dict):
        parent1 = res[n].parents[0]
        p = res[n].shape[0]
        return f"{n}[1:{p}] ~ ddirch(v{parent1._n}[1:{p}])"


    