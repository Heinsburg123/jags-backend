import subprocess
from pangolin.ir import RV, Add, Constant, Normal

jags_path = "C:/Program Files/JAGS/JAGS-4.3.1/x64/bin/jags.bat"

class Sample_prob:
    class RunDFS:
        def __init__(self):
            self.visited = {}
        
        def dfs(self, node):
            name = "v"+str(node._n)
            if name in self.visited:
                return
            self.visited[name] = node
            for parent in node.parents:
                self.dfs(parent)

        def run_dfs(self, nodes):
            for node in nodes:
                self.dfs(nodes[node])
            return self.visited 

    def sample(self, sample_vars:list[RV], kwargs:dict[RV, float|int]):
        dic = {}
        for var in kwargs:
            dic["v"+str(var._n)] = var
        for sample_var in sample_vars:
            dic["v"+str(sample_var._n)] = sample_var 
        app = self.RunDFS()
        res = app.run_dfs(dic)

        with open( "data.R", "w") as f:
            for var in kwargs:
                f.write(f"{("v"+str(var._n))} <- {kwargs[var]}\n")
            f.close()

        with open( "model.bug", "w") as f:
            f = open("model.bug", "w")
            f.write("model {\n")
            check = {}  
            for node in res: 
                if node in check:
                    continue
                check[node] = True
                if(type(res[node].op) == Constant):
                    f.write(f"{node} <- {res[node]}\n")
                if(type(res[node].op) == Add):
                    parent1 = None
                    parent2 = None
                    for node2 in res:
                        if res[node].parents[0] == res[node2]:
                            parent1 = str(node2)
                        if(res[node].parents[1] == res[node2]):
                            parent2 = str(node2)
                    f.write(f"{node} <- {parent1} + {parent2}\n") 
                if(type(res[node].op) == Normal):
                    parent1 = None 
                    parent2 = None
                    for node2 in res:
                        if res[node].parents[0] == res[node2]:
                            parent1 = str(node2)
                        if(res[node].parents[1] == res[node2]):
                            parent2 = str(node2)
                    f.write(f"{node} ~ dnorm({parent1}, {parent2})\n")     

            f.write("}\n")                  
            f.close()

        with open("script.txt", "w") as f:
            script = 'model in "model.bug"\n'
            script += "compile, nchains(1)\n"
            script += "initialize\n"
            script += "update 1000\n"
            for sample_var in sample_vars:
                script += f"monitor {('v'+str(sample_var._n))}\n"
            script += "update 1000\n"
            script += "coda *\n"
            f.write(script)

        cmd = f'"{jags_path}" script.txt'
        output = subprocess.check_output(cmd,  stderr=subprocess.STDOUT).decode()
        return output
    
    def read_coda(self):
        result = {};
        res_lines = [];
        with open("CODAchain1.txt", "r") as f:
            res_lines = [line.strip().split() for line in f.readlines()]
        for i in range(len(res_lines)):
            res_lines[i] = float(res_lines[i][1])
        with open("CODAindex.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                v, start, end  = line.strip().split()
                result[v] = res_lines[int(start)-1:int(end)];
        return result;


