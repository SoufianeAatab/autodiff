class Op():
    def __init__(self, op_name, a, b=None, z=None):
        self.op_name = op_name
        self.a = a
        self.b = b

    def __repr__(self):
        op_desc = self.op_name + '('
        op_desc += '%' + str(self.a.id) + '::' + str(self.a.shape)
        # if self.a.data is not None:
        #     op_desc += ' ' + str(self.a.data.shape)
        if self.b is not None:
            op_desc += ',%' + str(self.b.id) + '::' + str(self.b.shape)
        op_desc += ')'
        return op_desc
