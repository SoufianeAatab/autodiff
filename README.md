# Autodiff: Automatic Differentiation for Training Neural Networks on IoT Devices
During my thesis on "Training Neural Networks on IoT Devices," I faced challenges building neural network models in plain C and deploying them for on-device training. This involved creating operators and making sure they could calculate both forward and backward passes, handling inputs, and gradients from the loss function. It was tricky, and everything needed to fit just right. Since then, I've wanted to make a tool to automate this process. Recently, I found Andrej Karpathy's micrograd project on GitHub, which inspired me to start. After two weeks of work, I've managed to make it.

Implemented layers so far:
-----------------------
- Linear Layer
- Conv2d Layer
- Maxpool2d Layer
- LogSoftmax Layer
- NLL & Mse Loss Layers
- Activation Functions
    - Sigmoid
    - Relu
    - Tanh
 
Example usage:
-----------
main.py
```python
from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward

# Data placeholder
input = Param(None, var_name='input', shape=(1,28*28))
output = Param(None, var_name='output', shape=(1,10))

# Define model params, no need for setting data.
w1 = Param(None, shape=(32,28*28), var_name='w1')
w2 = Param(None, shape=(10,32), var_name='w2')

# forward pass
w_t = w1.t()
z = matmul(input,w_t)
a = sigmoid(z)
z2 = matmul(a, w2.t())
a2 = log_softmax(z2)
loss = nll_loss(a2, output)

# backward function receives the last operator of the model and its inputs
graph = backward(loss, [w1.id, w2.id])
ops = graph.build()

interpreter = Interpreter(ops, [w1, w2, input, output])
# gen C code
interpreter.gen_code()
param_grads = [w1.id, w2.id]
# generate sgd to update weights
interpreter.gen_sgd(grads, param_grads)
```

# Instructions
run 
```python main.py``` thiss command will generate and output the following code in c containg the forward + backward + sgd, ready to be used.
```c
float* buf = (float*)calloc(51828, sizeof(float));
mat_mul(&input_ptr[l * 784] /* (1, 784) */, &buf[0] /* (784, 32) */, &buf[26202] /* (1, 32) */, 1, 784, 784, 1, 784, 32, 1, 784); // (1, 32) 5
sigmoid(&buf[26202] /* (1, 32)*/ , &buf[26234] /*(1, 32)*/, 32); // (1, 32) 6
mat_mul(&buf[26234] /* (1, 32) */, &buf[25088] /* (32, 10) */, &buf[26266] /* (1, 10) */, 1, 32, 32, 1, 32, 10, 1, 32); // (1, 10) 8
log_softmax(&buf[26266], &buf[26276], 10); // (1, 10) 9
exp(&buf[26276], &buf[26286], 10); // (1, 10) 18
buf[26296] = nll_loss(&buf[26276], &y_ptr[l*10], 10); // (1, 10) 10
for(uint32_t k=0;k<10;++k){
    buf[26306+k] = 1;
    buf[26316+k] = -1; // (1, 10) 12
}
mul(&buf[26306], &buf[26316], &buf[26326], 10); // (1, 10) 16
mul(&buf[26326], &y_ptr[l*10], &buf[26336], 10); // (1, 10) 17
add(&buf[26286], &buf[26336], &buf[26346], 10); // (1, 10) 19
mat_mul(&buf[26346] /* (1, 10) */, &buf[25088] /* (10, 32) */, &buf[26356] /* (1, 32) */, 1, 10, 10, 1, 10, 32, 32, 1); // (1, 32) 23
sigmoid_diff(&buf[26202], &buf[26356], &buf[26388], 32); // (1, 32) 24
mat_mul(&buf[26388] /* (32, 1) */, &input_ptr[l * 784]  /* (1, 784) */, &buf[26420] /* (32, 784) */, 32, 1, 1, 32, 1, 784, 784, 1); // (32, 784) 27
mat_mul(&buf[26346] /* (10, 1) */, &buf[26234] /* (1, 32) */, &buf[51508] /* (10, 32) */, 10, 1, 1, 10, 1, 32, 32, 1); // (10, 32) 22

for (uint32_t k=0;k<25088;++k){
    buf[0 + k] -= buf[26420 + k] * lr;
}
for (uint32_t k=0;k<320;++k){
    buf[25088 + k] -= buf[51508 + k] * lr;
}
```
Feel free to try it out and contribute to the development effort!

