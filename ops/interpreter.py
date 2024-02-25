import torch

class Interpreter:
    def __init__(self, ops):
        self.ops = ops

    def run(self):
        for op in self.ops:
            if not op._op:
                continue
            print(f'var_{op.id}={op._op.get_inference_code()}')

    def gen_torch_code(self):
        for operator in self.ops:
            if not operator._op:
                continue
            op = operator._op
            if op.op_name == 'matmul':
                # op.data = a.data @ b.data
                print(f"var_{operator.id}=var_{op.a.id}@var_{op.b.id}")
            elif op.op_name == 'transpose':
                # op.data = a.data.t()
                print(f"var_{operator.id}=var_{op.a.id}.t()")
            elif op.op_name == 'sigmoid':
                # op.data = torch.nn.functional.sigmoid(a.data)
                print(f"var_{operator.id}=torch.nn.functional.sigmoid(var_{op.x.id})")
            elif op.op_name == 'mse':
                # a_tensor = a.data
                # b_tensor = b.data
                # assert a_tensor.shape == b_tensor.shape, f"mse two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                # op.data = torch.mean(torch.pow(a.data - b.data, 2), dim=-1, keepdim=True)
                print(f"var_{operator.id}=torch.nn.functional.mse_loss(var_{op.y.id}, var_{op.y_pred.id})")
            elif op.op_name == 'mul':
                # a_tensor = a.data
                # b_tensor = b.data
                # if not isinstance(a.data, torch.Tensor):
                #     a_tensor = torch.tensor(a.data).expand(a.shape)
                # if not isinstance(b.data, torch.Tensor):
                #     b_tensor = torch.tensor(b.data).expand(b.shape)
                # assert a_tensor.shape == b_tensor.shape, f"multipying two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                # op.data = a_tensor * b_tensor
                print(f"var_{operator.id}=var_{op.a.id} * var_{op.b.id}")
            elif op.op_name == 'add':
                # a_tensor = a.data
                # b_tensor = b.data
                # if not isinstance(a.data, torch.Tensor):
                #     a_tensor = torch.tensor(a.data).expand(a.shape)
                # if not isinstance(b.data, torch.Tensor):
                #     b_tensor = torch.tensor(b.data).expand(b.shape)
                # assert a_tensor.shape == b_tensor.shape, f"add two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                # op.data = a_tensor + b_tensor
                print(f"var_{operator.id}=var_{op.a.id} + var_{op.b.id}")

            elif op.op_name == 'sub':
                # a_tensor = a.data
                # b_tensor = b.data
                # if not isinstance(a.data, torch.Tensor):
                #     a_tensor = torch.tensor(a.data).expand(a.shape)
                # if not isinstance(b.data, torch.Tensor):
                #     b_tensor = torch.tensor(b.data).expand(b.shape)
                # assert a_tensor.shape == b_tensor.shape, f"add two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                # op.data = a_tensor - b_tensor
                print(f"var_{operator.id}=var_{op.a.id} - var_{op.b.id}")
            elif op.op_name == 'ones_like':
                # op.data = torch.ones_like(a.data)
                print(f"var_{operator.id}=torch.ones_like(var_{op.a.id})")
            elif op.op_name == 'sigmoid_diff':
                # op.data = torch.nn.functional.sigmoid(a.data) * (1.0-torch.nn.functional.sigmoid(a.data)) * b.data
                print(f"var_{operator.id}=torch.nn.functional.sigmoid(var_{op.x.id}) * (1.0-torch.nn.functional.sigmoid(var_{op.x.id})) * var_{op.grad.id}")
            elif op.op_name == 'mse_grad':
                # op.data = -2*b.data*a.data
                # print(f"var_{operator.id}=-2*(var_{b.id}-var_{a.id})")
                pass
            elif op.op_name == 'assign':
                print(f"var_{operator.id} = torch.tensor({op.a}).expand({operator.shape})")
            elif op.op_name == 'conv2d':
                print(f"var_{operator.id}=torch.nn.functiona.conv2d(var_{op.x.id}, var_{op.kernels.id}, kernel_size={op.kernel_size})")
            elif op.op_name == 'cross_entropy':
                print(f"var_{operator.id}=torch.nn.functiona.cross_entropy(var_{op.input.id}, var_{op.target.id})")
            elif op.op_name == 'log_softmax':
                print(f"var_{operator.id}=torch.nn.functiona.log_softmax(var_{op.x.id})")
            elif op.op_name == 'exp':
                print(f"var_{operator.id}=torch.exp(var_{op.a.id})")
            elif op.op_name == 'sum':
                print(f"var_{operator.id}=torch.sum(var_{op.x.id}, dim={op.dim})")
            elif op.op_name == 'reshape':
                print(f"var_{operator.id}=var_{op.a.id}.reshape({op.shape})")
            else:
                print(f"Op {op.op_name} not defined!")