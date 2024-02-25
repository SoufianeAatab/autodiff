import torch

class Interpreter:
    def __init__(self, ops):
        self.ops = ops

    def run(self):
        for op in self.ops:
            if not op._op:
                continue
            a = op._op.a
            b = op._op.b

            if op._op.op_name == 'matmul':
                # op.data = a.data @ b.data
                print(f"var_{op.id}=var_{a.id}@var_{b.id}")
            elif op._op.op_name == 'transpose':
                # op.data = a.data.t()
                print(f"var_{op.id}=var_{a.id}.t()")
            elif op._op.op_name == 'sigmoid':
                # op.data = torch.nn.functional.sigmoid(a.data)
                print(f"var_{op.id}=torch.nn.functional.sigmoid(var_{a.id})")
            elif op._op.op_name == 'mse':
                # a_tensor = a.data
                # b_tensor = b.data
                # assert a_tensor.shape == b_tensor.shape, f"mse two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                #op.data = torch.mean(torch.pow(a.data - b.data, 2), dim=-1, keepdim=True)
                print(f"var_{op.id}=torch.nn.functional.mse_loss(var_{a.id}, var_{b.id})")
            elif op._op.op_name == 'mul':
                # a_tensor = a.data
                # b_tensor = b.data
                # if not isinstance(a.data, torch.Tensor):
                    # a_tensor = torch.tensor(a.data).expand(a.shape)
                # if not isinstance(b.data, torch.Tensor):
                    # b_tensor = torch.tensor(b.data).expand(b.shape)
                # assert a_tensor.shape == b_tensor.shape, f"multipying two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                # op.data = a_tensor * b_tensor
                print(f"var_{op.id}=var_{a.id} * var_{b.id}")
            elif op._op.op_name == 'add':
                # a_tensor = a.data
                # b_tensor = b.data
                # if not isinstance(a.data, torch.Tensor):
                #     a_tensor = torch.tensor(a.data).expand(a.shape)
                # if not isinstance(b.data, torch.Tensor):
                #     b_tensor = torch.tensor(b.data).expand(b.shape)
                # assert a_tensor.shape == b_tensor.shape, f"add two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                # op.data = a_tensor + b_tensor
                print(f"var_{op.id}=var_{a.id} + var_{b.id}")

            elif op._op.op_name == 'sub':
                # a_tensor = a.data
                # b_tensor = b.data
                # if not isinstance(a.data, torch.Tensor):
                #     a_tensor = torch.tensor(a.data).expand(a.shape)
                # if not isinstance(b.data, torch.Tensor):
                #     b_tensor = torch.tensor(b.data).expand(b.shape)
                # assert a_tensor.shape == b_tensor.shape, f"add two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                # op.data = a_tensor - b_tensor
                print(f"var_{op.id}=var_{a.id} - var_{b.id}")
            elif op._op.op_name == 'ones_like':
                # op.data = torch.ones_like(a.data)
                print(f"var_{op.id}=torch.ones_like(var_{a.id})")
            elif op._op.op_name == 'sigmoid_grad':
                # op.data = torch.nn.functional.sigmoid(a.data) * (1.0-torch.nn.functional.sigmoid(a.data)) * b.data
                print(f"var_{op.id}=torch.nn.functional.sigmoid(var_{a.id}) * (1.0-torch.nn.functional.sigmoid(var_{a.id})) * var_{b.id}")
            elif op._op.op_name == 'mse_grad':
                # op.data = -2*b.data*a.data
                print(f"var_{op.id}=-2*(var_{b.id}-var_{a.id})")
            elif op._op.op_name == 'assign':
                print(f"var_{op.id} = torch.tensor({a.data}).expand({a.shape})")
            else:
                print(f"Op {op._op.op_name} not defined!")

    def gen_torch_code(self):
        pass