from scalar_ops import Scalar_ops
class flow:
    def VMap(n, res:dict):
        obj = res[n]
        op = res[n].op.base_op
        in_axis = res[n].op.in_axes
        axis_size = res[n].op.axis_size
        parents = obj.parents
        if(len(in_axis) != len(parents)):
            raise ValueError("Length of in_axes must be equal to number of parents")
        for i in range(len(in_axis)):
            if in_axis[i] is not None and axis_size is None:
                axis_size = parents[i].shape[0]
                break
        for i in range(len(in_axis)):
            if(in_axis[i] is not None and parents[i].shape[0] != axis_size):
                raise ValueError("All inputs with in_axes must have the same leading dimension")
            if(in_axis[i] is None and parents[i].ndim != 0):
                raise ValueError("All inputs without in_axes must have only 0 leading dimension")
        ans =""
        ans += f"for(i in 1:{axis_size})" + "{\n"
        code = Scalar_ops.__dict__[op.name](n, res)
        for i in range(len(parents)):
            if in_axis[i] is not None:
                code = code.replace(f"v{parents[i]._n}", f"v{parents[i]._n}[i]")
        code = code.replace(n, f"{n}[i]")
        ans += "  " + code + "\n}"
        return ans
