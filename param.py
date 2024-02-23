from ops.op import Op

grads = {}
global_id = -1

class Param():
    def __init__(self, x, children = (), shape = (1,1), var_name="", print_init = False):  
        global global_id
        global_id += 1
        self.data = x
        self.id = global_id
        self.other = None
        self.var_name = var_name
        self.shape = shape

        self._prev = set(children)
        self._op = None
        self._backward_op = None
        self._backward = None

        if print_init:
            print(self)
    
    def t(self):
        z = Param(None, children=(self,), shape=(self.shape[1], self.shape[0]), 
                  var_name=self.var_name + '_t', 
                  print_init=False)
        
        op = Op('transpose', self, b=None)
        z._op = op
        def backward():
            grads[op.a.id] = grads[z.id]
            # print('t::', self.id, z.id, op)
            return op.a
        z._backward = backward
        return z

    def compile(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        # print(visited)
        ordered_ops = []
        for v in topo:
            if v._op is not None:
                ordered_ops.append(v._op)
        return ordered_ops
    
    def build(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        return topo

    def get_data(self):
        return self.data
    
    def __repr__(self):
        return f"%{self.id}=" + str(self._op) + f' shape={self.shape}'

