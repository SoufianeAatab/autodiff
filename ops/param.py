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

    def __rmul__(self, other):
        if not isinstance(other, Param):
            other = Param(other, children=(), shape=self.shape)
            other._op = Op('assign', other)
            return self.__mul__(other)

    def __mul__(self, other):
        z = Param(None, children=(self, other), shape=self.shape)
        op = Op('mul', self, b = other)
        z._op = op
        def backward():
            aux = grads[op.b.id]
            grads[op.b.id] = grads[op.a.id]
            grads[op.a.id] = aux
        z._backward = backward
        return z
    
    def __add__(self, other):
        z = Param(None, children=(self, other), shape=self.shape)
        op = Op('add', self, b = other)
        z._op = op
        def backward():
            grads[op.b.id] = grads[z.id]
            grads[op.a.id] = grads[z.id]
        z._backward = backward
        return z
    
    def __sub__(self, other):
        z = Param(None, children=(self, other), shape=self.shape)
        op = Op('sub', self, b = other)
        z._op = op
        def backward():
            grads[op.b.id] = -grads[z.id]
            grads[op.a.id] = grads[z.id]
        z._backward = backward
        return z
    
    def __neg__(self):
        return self * -1

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
        if self._op is not None:
            return f"%{self.id}=" + str(self._op) + f' shape={self.shape}'
        else:
            return f"%{self.id}::shape={self.shape}"

