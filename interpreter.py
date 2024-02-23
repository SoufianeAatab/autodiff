import torch


# torch inference engine
def run(ops):
    for op in ops:
        if op.op_name == 'matmul':
            op.z.data = op.a.data @ op.b.data
            # op.z.grad.data = torch.zeros_like(op.z.data)
        elif op.op_name == 'linear':
            op.z.data = op.a.data @ op.b.data.t()
            # op.z.grad.data = torch.zeros_like(op.z.data)
        elif op.op_name == 'transpose':
            op.z.data = op.a.data.t()
            # op.z.grad.data = torch.zeros_like(op.z.data)
        elif op.op_name == 'sigmoid':
            op.z.data = 1.0 / (1.0 + torch.exp(-op.a.data))
            # op.z.grad.data = torch.zeros_like(op.z.data)
        elif op.op_name == 'mse':
            op.z.data = torch.mean(torch.pow(op.a.data - op.b.data, 2))
            # op.z.grad.data = torch.zeros_like(op.z.data)
        elif op.op_name == 'add':
            assert op.a.data.shape == op.b.data.shape, "Adding two different shape tensors"
            op.z.data = op.a.data + op.b.data
            # op.z.grad.data = torch.zeros_like(op.z.data)
        else:
            print(f"Op {op.op_name} not defined!")

