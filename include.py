import subprocess
from pangolin.ir import *
from  Backend.scalar_ops import Scalar_ops
from Backend.Multi_funcs import Multi_funcs
from Backend.flow import flow
from Backend.index import index
import platform
import re

def ensure_size( arr, sizes, depth=0):
    while len(arr) < sizes[depth]:
        if depth == len(sizes) - 1:
            arr.append(None)    # leaf
        else:
            arr.append([])
    return arr


class Sample_prob:
    calculate_value = {}    
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
            for node in res:
                if(res[node].op.name == "Constant"):
                    f.write(Scalar_ops.__dict__["Constant_before"](node, res))
            for var in kwargs:
                f.write(f"{("v"+str(var._n))} <- {kwargs[var]}\n")
            f.close()
        
        with open( "model.bug", "w") as f:
            f.write("model {\n")
            check = {}  
            for node in sorted(res): 
                if node in check:
                    continue
                check[node] = True
                parents = [f"v{res[node].parents[i]._n}" for i in range(len(res[node].parents))]
                if(flow.__dict__.get(res[node].op.name) is not None):
                    tmp = [res[node].parents[i] for i in range(len(res[node].parents))]
                    code = flow.__dict__[res[node].op.name](node, res[node].op, parents,0, tmp)
                    f.write(code + "\n")
                elif(index.__dict__.get(res[node].op.name) is not None):
                    tmp = index()
                    tmpp = [res[node].parents[i] for i in range(len(res[node].parents))]
                    code = tmp.SimpleIndex(node, parents, tmpp)
                    f.write(code + "\n")
                elif(res[node].op.name!="Constant" and Scalar_ops.__dict__.get(res[node].op.name) is not None):
                    code = Scalar_ops.__dict__[res[node].op.name](node, parents)
                    f.write(code + "\n")
                elif(Multi_funcs.__dict__.get(res[node].op.name) is not None):
                    if(res[node].op.name == "Sum" or res[node].op.name == "Inv" or res[node].op.name == "Matmul"):
                        code = Multi_funcs.__dict__[res[node].op.name](node, res, self.calculate_value)
                    else:
                        code = Multi_funcs.__dict__[res[node].op.name](node, res)
                    f.write(code + "\n")
            f.write("}\n")                  
            f.close()

        with open("script.txt", "w") as f:
            script = 'model in "model.bug"\n'
            script += 'data in "data.R"\n'
            script += "compile, nchains(1)\n"
            script += "initialize\n"
            script += "update 1000\n"
            for sample_var in sample_vars:
                script += f"monitor {('v'+str(sample_var._n))}\n"
            script += "update 2000\n"
            script += "coda *\n"
            f.write(script)

        system = platform.system()
        if system == "Windows":
            jags_path = "C:/Program Files/JAGS/JAGS-4.3.1/x64/bin/jags.bat"
            cmd = f'"{jags_path}" script.txt'
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode()
        else:  # Linux/macOS
            cmd = ['jags', 'script.txt']
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        
        return(output)


    def read_coda(self):
        result = {}

        # Load MCMC samples
        with open("CODAchain1.txt", "r") as f:
            res_lines = [float(line.strip().split()[1]) for line in f]

        # Regex for v123[1,2,3] or v5 or v12[4]
        pattern = re.compile(r'(v\d+)(?:\[(.*?)\])?')

        with open("CODAindex.txt", "r") as f:
            for line in f:
                v, start, end = line.strip().split()

                match = pattern.fullmatch(v)
                name = match.group(1)
                index_str = match.group(2)

                # Slice values
                values = res_lines[int(start)-1 : int(end)]

                # Case 1: scalar (no indices)
                if index_str is None:
                    result[name] = values
                    continue

                # Case 2: multi-dimensional variable
                indices = list(map(int, index_str.split(",")))

                # Create variable if first time
                if name not in result:
                    result[name] = []

                arr = result[name]

                # Ensure the list is large enough
                arr = ensure_size(arr, indices)

                # Navigate to the leaf
                ref = arr
                for d in range(len(indices)-1):
                    idx = indices[d] - 1
                    ref[idx] = ensure_size(ref[idx], indices, depth=d+1)
                    ref = ref[idx]

                # Set value at final index
                ref[indices[-1] - 1] = values

                result[name] = arr
        return result


