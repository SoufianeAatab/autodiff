from ops.op import Op
from ops.ops import Add, Mul, Sub, Transpose, Assign, Reshape, Exp

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

        strides = []
        list_shape = list(shape)
        for s in reversed(range(len(list_shape))):
            l = 1
            for i in range(1,s+1):
                l *= list_shape[i]
            strides.append(l)

        #print(strides)
        self.stride = tuple(strides)

    def t(self):
        z = Param(None, children=(self,), shape=(self.shape[1], self.shape[0]))
        z.stride = (self.stride[1], self.stride[0])
        op = Transpose(self) # Op('transpose', self, b=None)
        z._op = op
        def backward():
            grads[op.a.id] = grads[z.id]
            # print('t::', self.id, z.id, op)
        z._backward = backward
        return z
    
    def exp(self):
        z = Param(None, children=(self,), shape=(self.shape[0], self.shape[1]))        
        op = Exp(self)
        z._op = op
        def backward():
            grads[op.a.id] = grads[z.id]
            # print('t::', self.id, z.id, op)
        z._backward = backward
        return z

    def __rmul__(self, other):
        if not isinstance(other, Param):
            out = Param(other, children=(), shape=self.shape)
            out._op = Assign(other)
            return self.__mul__(out)

    def __mul__(self, other):
        if not isinstance(other, Param):
            out = Param(other, children=(), shape=self.shape)
            out._op = Assign(other)
            other = out

        z = Param(None, children=(self, other), shape=self.shape)
        op = Mul(self, other)
        z._op = op
        def backward():
            aux = grads[op.b.id]
            grads[op.b.id] = grads[op.a.id]
            grads[op.a.id] = aux
        z._backward = backward
        return z
    
    def __add__(self, other):
        z = Param(None, children=(self, other), shape=self.shape)
        op = Add(self, other)
        #Op('add', self, b = other)
        z._op = op
        def backward():
            grads[op.b.id] = grads[z.id]
            grads[op.a.id] = grads[z.id]
        z._backward = backward
        return z
    
    def __sub__(self, other):
        z = Param(None, children=(self, other), shape=self.shape)
        op = Sub(self, other)
        #Op('sub', self, b = other)
        z._op = op
        def backward():
            grads[op.b.id] = -grads[z.id]
            grads[op.a.id] = grads[z.id]
        z._backward = backward
        return z
    
    def __neg__(self):
        from ops.functional import const
        return self * const(-1, self.shape)
    
    def reshape(self, shape):
        assert isinstance(shape, tuple), "Expering shape to be a tuple not a int"
        z = Param(None, children=(self,), shape=shape)
        op = Reshape(self, shape) # Op('transpose', self, b=None)
        z._op = op
        def backward():
            grads[op.a.id] = grads[z.id].reshape(self.shape)
        z._backward = backward
        return z

    def build(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v is not None: 
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
        build_topo(self)
        return topo

    def get_data(self):
        return self.data
    
    def __repr__(self):
        if self._op is not None:
            return f"%{self.id}=" + str(self._op.op_name) + f' shape={self.shape} value=' + (str(self.data)) if self.data is not None else f"%{self.id}=" + str(self._op.op_name) + f' shape={self.shape}'
        else:
            return f"%{self.id}::shape={self.shape}"

