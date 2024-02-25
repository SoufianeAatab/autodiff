from ops.op import Op
class Matmul(Op):
    def __init__(self, a, b):
        super().__init__('matmul')
        self.a = a
        self.b = b

    def __repr__(self):
        op_desc = self.op_name + '('
        op_desc += '%' + str(self.a.id) + '::' + str(self.a.shape)
        if self.b is not None:
            op_desc += ',%' + str(self.b.id) + '::' + str(self.b.shape)
        op_desc += ')'
        return op_desc
