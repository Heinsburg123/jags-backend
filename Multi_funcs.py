
class Matrix_funcs:
    def matmul(n, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} <- v{parent1._n} %*% v{parent2._n}"

    def transpose(n, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- t(v{parent1._n})"

    def inverse(n, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- solve(v{parent1._n})"
    
    def Sum(n, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];  
        if(parent2.op.value == 1):
            return f"{n} <- rowSums(v{parent1._n})"