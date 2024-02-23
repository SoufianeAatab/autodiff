from ops.param import grads, Param
def backward(output, vars):
    """ 
    Add the gradient of loss fn wrt to itself. 
    To connect the graph, is important to add the grad of the loss wrt to itself as a children.
    """
    from ops.functional import ones_like

    #Param(None, (output,), output.shape, var_name="grad_loss_wrt_loss", print_init=False)
    grad = ones_like(output)
    grad._backward = None
    grads[output.id] = grad
    output = grad

    # topological order all of the children in the graph    
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                # print('CHILD:::',child, 'parent::', v.id)
                build_topo(child)
            topo.append(v)

    build_topo(output)
    ops = []
    for v in reversed(topo):
        if isinstance(v, Param) and v._backward:
            v._backward()
            if v._op.b is not None:
                ops.extend([v._op.a, v._op.b])
            else:
                ops.append(v._op.a)

    gradients_for = []
    for var in vars:
        gradients_for.append(grads[var])

    return Param(None, children=(gradients_for))
