# Autodiff: Automatic Differentiation for Neural Networks in C

An automatic differentiation framework that generates standalone C code for training neural networks. The backward pass is computed during code generation rather than at runtime.

## Background

This project emerged from my thesis on "Training Neural Networks on IoT Devices," where implementing forward and backward passes for each operator manually was time-consuming and error-prone. This tool automates that process by generating C code from a Python model definition.

The generated C code has minimal dependencies and can be compiled for embedded devices. Custom operators can be implemented separately without recomputing the backward graph.

## Implemented Operators

## Implemented Operators

**Layers:**
- Linear
- Conv2d
- MaxPool2d
- LogSoftmax
- NLL Loss, MSE Loss

**Activations:**
- Sigmoid
- ReLU
- Tanh

## Usage

## Usage

Define a model using the Python API:

```python
from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward

# Data placeholders
input = Param(None, var_name='input', shape=(1, 28*28))
output = Param(None, var_name='output', shape=(1, 10))

# Model parameters
w1 = Param(None, shape=(32, 28*28), var_name='w1')
w2 = Param(None, shape=(10, 32), var_name='w2')

# Forward pass
z = matmul(input, w1.t())
a = sigmoid(z)
z2 = matmul(a, w2.t())
a2 = log_softmax(z2)
loss = nll_loss(a2, output)

# Build backward graph
graph = backward(loss, [w1.id, w2.id])
ops = graph.build()

# Generate C code
interpreter = Interpreter(ops, [w1, w2, input, output])
interpreter.gen_code()
interpreter.gen_sgd(grads, [w1.id, w2.id])
```

Run:
```bash
python main.py
```

See [`main.py`](main.py) for complete example.

## Generated Code

Example output:

```c
float* buf = (float*)calloc(51828, sizeof(float));

// Forward pass
mat_mul(&input_ptr[l * 784], &buf[0], &buf[26202], 1, 784, 784, 1, 784, 32, 1, 784);
sigmoid(&buf[26202], &buf[26234], 32);
mat_mul(&buf[26234], &buf[25088], &buf[26266], 1, 32, 32, 1, 32, 10, 1, 32);
log_softmax(&buf[26266], &buf[26276], 10);
buf[26296] = nll_loss(&buf[26276], &y_ptr[l*10], 10);

// Backward pass
mul(&buf[26306], &buf[26316], &buf[26326], 10);
// ... gradient computation ...

// SGD update
for (uint32_t k=0; k<25088; ++k) {
    buf[0 + k] -= buf[26420 + k] * lr;
}
```

The generated code can be compiled with any C compiler.

## Testing

Gradients are validated against PyTorch to ensure correctness of the backward pass.

## Installation

```bash
git clone <your-repo-url>
cd autodiff
pip install -r requirements.txt  # PyTorch, only needed for testing
```

## Notes

This is a personal side project. The basic operators work but the codebase is not extensively tested or optimized.

## Acknowledgments

Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy.
