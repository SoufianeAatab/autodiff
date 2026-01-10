from ops.ops import Add, Mul, Sub, Transpose, Assign, Reshape, Exp

grads = {}
global_id = -1

class Param():
    def __init__(self, x, children = (), shape = (1,1), require_grads=True, var_name="", print_init = False):  
        global global_id
        global_id += 1
        self.data = x
        self.id = global_id
        self.other = None
        self.var_name = var_name
        self.shape = shape
        self.require_grads = require_grads

        self._prev = set(children)
        self._backward_op = None
        self._backward = None

        if self.data is not None:   
            self._op = Assign(self, data=self.data)
        else:
            self._op = None

        strides = []
        list_shape = list(shape)
        for s in reversed(range(len(list_shape))):
            l = 1
            for i in range(1,s+1):
                l *= list_shape[i]
            strides.append(l)

        # print(f"Strides for {var_name} are {strides}")
        
        self.stride = tuple(strides)
    
    # For transpose we just switch the strides and shape
    def t(self):
        """
        Transposes the current parameter by swapping its shape and strides.
        Returns a new Param object representing the transposed matrix.
        """
        # Create a new Param object with swapped shape (rows <-> columns)
        z = Param(None, children=(self,), shape=(self.shape[1], self.shape[0]))
        z.stride = (self.stride[1], self.stride[0])
        # Create a Transpose operation to track this operation in the computation graph
        op = Transpose(self) # Op('transpose', self, b=None)
        z._op = op
        def backward():
            # The gradient of the input is the gradient of the output (transpose is its own inverse)
            grads[op.a.id] = grads[z.id]
        z._backward = backward
        return z
    
    def exp(self):
        z = Param(None, children=(self,), shape=(self.shape[0], self.shape[1]))        
        op = Exp(self)
        z._op = op
        # TODO: Is this backward function correct?
        def backward():
            grads[op.a.id] = grads[z.id]
        z._backward = backward
        return z

    # def __rmul__(self, other):
    #     if not isinstance(other, Param):
    #         out = Param(other, children=(), shape=self.shape)
    #         out._op = Assign(other)
    #         return self.__mul__(out)

    def __mul__(self, other):
        if not isinstance(other, Param):
            other = Param(other, children=(), shape=self.shape)
            # out._op = Assign(other)
            # other = out
        assert self.shape == other.shape, "Mismatch shape in * operator"

        z = Param(None, children=(self, other), shape=self.shape)
        op = Mul(self, other)
        z._op = op
        # z = a * b
        # da/dz = b * grad
        # db/dz = a * grad
        def backward():
            grad_z = grads[z.id]
            grads[op.a.id] = other * op.b
            grads[op.b.id] = self * op.a
            # aux = grads[op.b.id]
            # grads[op.b.id] = grads[op.a.id]
            # grads[op.a.id] = aux
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
        child = ""
        for c in list(self._prev):
            if c:
                child += f"%{c.id}, "

        child = child[:-2]
        if self._op is not None:
            return f"%{self.id} = {str(self._op.op_name)}({child}) shape={self.shape}" 
        else:
            return f"%{self.id}::shape={self.shape}"

