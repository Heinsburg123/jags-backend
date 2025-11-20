from scalar_ops import Scalar_ops
from Multi_funcs import Multi_funcs
from pangolin.ir import RV
class flow:
    def VMap(n, res:dict):
        op = res[n].op.base_op
        in_axes = res[n].op.in_axes
        axis_size = res[n].op.axis_size
        parents = res[n].parents
        if(len(in_axes) != len(parents)):
            raise ValueError("Length of in_axes must be equal to number of parents")
        for i in range(len(in_axes)):
            if in_axes[i] is not None and axis_size is None:
                axis_size = parents[i].shape[0]
                break
        for i in range(len(in_axes)):
            if(in_axes[i] is not None and parents[i].ndim == 0):
                raise ValueError("Input should be a vector if in_axes is not None")
            elif(in_axes[i] is not None and parents[i].shape[0] != axis_size):
                raise ValueError("All inputs with in_axes must have the same leading dimension")
            elif(in_axes[i] is None and parents[i].ndim != 0):
                raise ValueError("All inputs without in_axes must have only 0 leading dimension")
        ans =""
        ans += f"for(i in 1:{axis_size})" + "{\n"
        code = Scalar_ops.__dict__[op.name](n, res)
        for i in range(len(parents)):
            if in_axes[i] is not None:
                code = code.replace(f"v{parents[i]._n}", f"v{parents[i]._n}[i]")
        code = code.replace(n, f"{n}[i]")
        ans += "  " + code + "\n}"
        return ans

    def Autoregressive(n, res:dict):
        op = res[n].op.base_op
        length = res[n].op.length
        in_axes = res[n].op.in_axes
        where_self = res[n].op.where_self
        parents = res[n].parents
        if(len(in_axes) != len(parents)-1):
            raise ValueError("Length of in_axes must be equal to number of parents")
        if(where_self < 0 or where_self >= len(parents)):
            raise ValueError("The position of self should be in correct range")
        if(parents[where_self].ndim != 0):
            raise ValueError("Only an initial constant for autoregressive parent")
        offset = 0
        for i in range(len(parents)):
            if i == where_self:
                offset+=1
                continue
            if(in_axes[i-offset] is not None and parents[i].ndim==0):
                raise ValueError("Should have in_axes as None if parent is a single value constant")
            elif(in_axes[i-offset] is None and parents[i].ndim != 0):
                raise ValueError("Should have in_axes as 0 if parent is a constant vector")
            elif(in_axes[i-offset] is not None and parents[i].shape[0] != length):
                raise ValueError("Length of vector should match length of autoregressive")
        ans = f"{Scalar_ops.__dict__[op.name](n, res)}\n"
        ans += f"for(i in 2:{length})" + "{\n"
        code = Scalar_ops.__dict__[op.name](n, res)
        offset = 0
        for i in range(len(parents)):
            if(i == where_self):
                code = code.replace(f"v{parents[i]._n}", f"{n}[i-1]")
                ans = ans.replace(n, f"{n}[1]")
                offset+=1
                continue
            elif in_axes[i-offset] is not None:
                ans = ans.replace(f"v{parents[i]._n}", f"v{parents[i]._n}[1]")
                code = code.replace(f"v{parents[i]._n}", f"v{parents[i]._n}[i]")
        code = code.replace(f"{n}", f"{n}[i]", 1)
        ans += "  " + code + "\n}"
        return ans

    def Composite(n, res:dict):
        num = res[n].op.num_inputs
        ops = res[n].op.ops
        par_nums = res[n].op.par_nums
        parents = res[n].parents
        if(len(par_nums) != len(ops)):
            raise ValueError("number of ops should match the number of par_nums")
        if(num != len(parents)):
            raise ValueError("The number of parents should match num_inputs")
        new_list = []
        ans = ""
        for i in range(len(par_nums)):
            args = [ops[i]]
            for j in range(len(par_nums[i])):
                if(par_nums[i][j] < num):
                    args.append(parents[par_nums[i][j]])
                else:
                    if(par_nums[i][j]-num >= len(new_list)):
                        raise ValueError("Can't take parent that hasn't been created")
                    args.append(new_list[par_nums[i][j]-num])
            new_res = {}
            v = RV(*args)
            if(i<len(par_nums)-1):
                new_res[f"{n}_{i+1}"] = v
                code = Scalar_ops.__dict__[ops[i].name](f"{n}_{i+1}", new_res)
                for j in range(len(v.parents)):
                    if(par_nums[i][j] >= num):
                        code = code.replace(f"v{v.parents[j]._n}", f"{n}_{par_nums[i][j]-num+1}")
                new_list.append(v)
                ans+=code + "\n"
            else:
                new_res[n] = v
                code = Scalar_ops.__dict__[ops[i].name](n, new_res)
                for j in range(len(v.parents)):
                    if(par_nums[i][j] >= num):
                        code = code.replace(f"v{v.parents[j]._n}", f"{n}_{par_nums[i][j]-num+1}")
                ans+=code + "\n"
        
        return ans
                