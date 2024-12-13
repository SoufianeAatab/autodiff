from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward

def head(x, embs, head_size):
    query = Param(None, var_name="query", shape=(embs, head_size))
    values = Param(None, var_name="query", shape=(embs, head_size))
    key = Param(None, var_name="query", shape=(embs, head_size))
    
    q_t = query.t()
    k_t = key.t()
    v_t = values.t()

    q = matmul(q_t, x)
    k = matmul(k_t, x)
    wei = matmul(q, k)
    wei_tril = tril(wei)
    wei = log_softmax(wei_tril)
    v = matmul(v_t, x)
    return matmul(v, wei)


def mnist_example():
    # Data placeholder
    vocab = "abcdefghijklmnopqrstuvwxyz"
    vocab_size = len(vocab)
    seq_length = 8
    embs_size = 32
    input = Param(None, var_name='input', shape=(1, seq_length))
    output = Param(None, var_name='output', shape=(1,vocab_size))

    embs = Param(None, shape=(embs_size, vocab_size), var_name='w1', print_init=True)
    pos = Param(None, shape=(embs_size, seq_length), var_name='w2', print_init=True)

    embs_w_t = embs.t()
    z = matmul(input,embs_w_t)
    seq = Param(list(range(seq_length)), shape=(1, seq_length))
    z1 = matmul(seq,pos)
    z = z + z1
    head(z, embs_size, 8)

    