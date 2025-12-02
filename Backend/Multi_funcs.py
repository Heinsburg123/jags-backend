import numpy as np
from pangolin.ir import *
from  Backend.scalar_ops import Scalar_ops

class Multi_funcs:
    def Matmul(n, op, parents, res):
        return f"{n} <- v{parents[0]} %*% v{parents[1]}\n"

    def Sum(n, op, parents, res):
        code = ""
        offset = 0
        for i in range(res[0].ndim):
            if(i!=op.axis):
                code += f"for (j{i-offset} in 1:{res[0].shape[i]})"+"{\n"
            else:
                offset+=1
        code+=n
        for i in range(res[0].ndim-1):
            if(code[-1]==']'):
                code = code[:-1]+f",j{i}]"
            else:
                code = code + f"[j{i}]"
        code+= f"<- sum({parents[0]}"
        offset = 0
        for i in range(res[0].ndim):
            if(code[-1]==']'):
                if(i == op.axis):
                    code = code[:-1]+",]"
                    offset+=1
                else:
                    code = code[:-1]+f",j{i-offset}]"
            else:
                if(i == op.axis):
                    code = code+f"[]"
                    offset+=1
                else:
                    code = code+f"[j{i-offset}]"
        if(code[-1]!=']'):
            code+="[]"
        code+=")\n"
        for i in range(res[0].ndim-1):
            code +="}"
        return code
    def Softmax(n, op, parents, res):
        k = res[0].shape[0]
        idd = n.find('[')
        if(idd == -1):
            name1 = f"{n}_1"
            name2 = f"{n}_2"
        else:
            name1 = n[:idd]+f"_1"+n[idd:]
            name2 = n[:idd]+f"_2"+n[idd:]
        code = ""
        code += f"for (i in 1:{k})"+"{\n"
        code += f"  {name1}[i] <-exp(v{parents[0]}[i])\n"+"}\n"
        code += f"{name2} <- sum({name1}[])\n"
        code += f"for (i in 1:{k})" + "{\n"
        code += f"  {n}[i] <- {name1}[i]/{name2}\n" + "}\n"
        return code
    
    def MultiNormal(n, op, parents, res):
        p = res[0].shape[0]
        return f"{n}[1:{p}] ~ dmnorm(v{parents[0]}[1:{p}], inverse(v{parents[1]}[1:{p},1:{p}]))"
    
    def Multinomial(n, op, parents, res):
        p = res[1].shape[0]
        return f"{n}[1:{p}] ~ dmulti(v{parents[1]}[1:{p}], v{parents[0]._n})"

    def Dirichlet(n, op, parents, res):
        p = res[0].shape[0]
        return f"{n}[1:{p}] ~ ddirch(v{parents[0]}[1:{p}])"


    