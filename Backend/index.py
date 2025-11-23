class index:
    code = ""
    def loop(self, name, cur, array, arr_name, index):
        if(len(cur) == len(array)):
            self.code += name +"["
            tmp = "["
            for i in range(len(index)):
                self.code +=f"{index[i]+1},"
                tmp += f"{cur[i]+1},"
            self.code = self.code[:-1] + "]"
            tmp = tmp[:-1] + "]"
            self.code += f"<- {arr_name}{tmp}\n"
        else:
            for i in range (len(array[len(cur)])):
                x = array[len(cur)][i]
                self.loop(name, cur+[x], array, arr_name, index + [i])
    
    def SimpleIndex(self, n, res:dict):
        parent = res[n].parents[0]
        indexes = []
        for i in range(1, len(res[n].parents)):
            indexes.append(res[n].parents[i].op.value)
        array = parent.op.value 
        self.loop(n, [], indexes, f"v{parent._n}", [])
        ans = self.code
        self.code = ""
        return ans
