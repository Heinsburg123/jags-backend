import numpy as np 

class Scalar_ops:
    def Constant(n:str, res:dict):
        code = ""
        if res[n].ndim == 0:
            code += f"{n} <- {res[n].op.value}\n"
        else:
            code += f"{n}<-structure(c("
            arr = np.array(res[n].op.value)
            flat = arr.reshape(-1)
            for i in range(len(flat)):
                if(i == len(flat)-1):
                    code += f"{flat[i]}"
                else:
                    code += f"{flat[i]},"
            code += f"),.Dim=c("
            for i in range(res[n].ndim):
                if(i == res[n].ndim-1):
                    code += f"{res[n].shape[i]}))\n"
                else:
                    code += f"{res[n].shape[i]},"
        return code

    def Add(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} <- v{parent1._n} + v{parent2._n}"

    def Sub(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} <- v{parent1._n} - v{parent2._n}"
    
    def Mul(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} <- v{parent1._n} * v{parent2._n}"
    
    def Div(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} <- v{parent1._n} / v{parent2._n}"
    
    def Pow(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} <- v{parent1._n} ^ v{parent2._n}"
    

    def Normal(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} ~ dnorm(v{parent1._n}, 1/(v{parent2._n}*v{parent2._n}))"

    def Cauchy(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} ~ dt(v{parent1._n}, 1/(v{parent2._n}*v{parent2._n}), 1)"
    
    def NormalPrec(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} ~ dnorm(v{parent1._n}, v{parent2._n})"
    
    def Lognormal(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} ~ dlnorm(v{parent1._n}, 1/(v{parent2._n}*v{parent2._n}))"
    
    def Bernoulli(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} ~ dbern(v{parent1._n})"
    
    def BetaBinomial(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        parent3 = res[n].parents[2];
        return f"{n} ~ dbin({n}_5, v{parent1._n})\n{n}_5 ~ dbeta(v{parent2._n}, v{parent3._n})"
    
    def BernoulliLogit(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} ~ dbern({n}_5)\nlogit({n}_5) <- v{parent1._n}"
    
    def Binomial(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} ~ dbin(v{parent1._n}, v{parent2._n})"
    
    def Uniform(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} ~ dunif(v{parent1._n}, v{parent2._n})"
    
    def Categorical(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} ~ dcat(v{parent1._n})"

    def Beta(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} ~ dbeta(v{parent1._n}, v{parent2._n})"

    def Exponential(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} ~ dexp(v{parent1._n})"
    
    def Gamma(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        return f"{n} ~ dgamma(v{parent1._n}, v{parent2._n})"
    
    def Poisson(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} ~ dpois(v{parent1._n})"
    
    def StudentT(n:str, res:dict):
        parent1 = res[n].parents[0];
        parent2 = res[n].parents[1];
        parent3 = res[n].parents[2];
        return f"{n} ~ dt(v{parent1._n}, 1/(v{parent2._n}*v{parent2._n}), v{parent3._n})"
    
    def Abs(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- abs(v{parent1._n})"
    
    def Arccos(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- acos(v{parent1._n})"
    
    def Arcsin(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- asin(v{parent1._n})"
    
    def Arccosh(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- acosh(v{parent1._n})"
    
    def Arcsinh(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- asinh(v{parent1._n})"
    
    def Arctan(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- atan(v{parent1._n})"
    
    def Arctanh(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- atanh(v{parent1._n})"
    
    def Cos(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- cos(v{parent1._n})"
    
    def Sin(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- sin(v{parent1._n})"
    
    def Tan(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- tan(v{parent1._n})"
    
    def Cosh(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- cosh(v{parent1._n})"
    
    def Sinh(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- sinh(v{parent1._n})"
    
    def Tanh(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- tanh(v{parent1._n})"

    def Exp(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- exp(v{parent1._n})"
    
    def Log(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- log(v{parent1._n})"
    
    def Loggamma(n:str, res:dict):
        parent1 = res[n].parents[0]
        return f"{n} <- loggam(v{parent1._n})"
    
    def InvLogit(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- exp(v{parent1._n})/(1 + exp(v{parent1._n}))"
    
    def Logit(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- log(v{parent1._n}/(1 - v{parent1._n}))"
    
    def Step(n:str, res:dict):
        parent1 = res[n].parents[0];
        return f"{n} <- step(v{parent1._n})"
    
    