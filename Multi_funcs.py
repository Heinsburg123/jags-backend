class Multi_funcs:
    def MatMul(n, res:dict):
        parent1 = res[n].parents[0]
        parent2 = res[n].parents[1]
        k = parent1.shape[0]
        m = parent1.shape[1]
        p = parent2.shape[1]
        return f"for (i in 1:{k})" + "{\n" + f"for (j in 1:{p})" + "{\n" + f"{n}[i,j]<-inprod(v{parent1._n}[i, 1:{m}], v{parent2._n}[1:{m}, j])\n"+"}\n"+"}\n"

    def Inv(n, res:dict):
        parent1 = res[n].parents[0]
        return f"{n} <- solve(v{parent1._n})"
    
    def Softmax(n, res:dict):
        parent1 = res[n].parents[0]
        return f"{n} <- exp(v{parent1._n}) / sum(exp(v{parent1._n}))"
    
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


    