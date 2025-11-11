class Matrix_funcs:
    def matmul(n, res:dict):
        parent1 = res[n].parents[0]
        parent2 = res[n].parents[1]
        return f"{n} <- v{parent1._n} %*% v{parent2._n}"

    def transpose(n, res:dict):
        parent1 = res[n].parents[0]
        return f"{n} <- t(v{parent1._n})"

    def inverse(n, res:dict):
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


    