from ops.param import grads, Param
def backward(output, vars):
    """ 
    Add the gradient of loss fn wrt to itself. 
    To connect the graph, is important to add the grad of the loss wrt to itself as a children.
    """
    from ops.functional import ones_like

    grad = ones_like(output)
    grad._backward = None
    grads[output.id] = grad
    output = grad

    # topological order of the children in the graph    
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(output)
    ops = []
    for v in reversed(topo):
        if isinstance(v, Param) and v._backward:
            v._backward()
            ops.extend([list(v._op.child)])
     
    gradients_for = []
    for var in vars:
        gradients_for.append(grads[var])

    return Param(None, children=(gradients_for))
