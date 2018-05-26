## Define-by-Run

All define-by-run frameworks build a computation graph at every iterations. The computation graph doesn't contain any control flow logic because "AD is blind with respect to any operation, including control flow statements, which do not directly alter numeric values."[1] 

### Overview

| Framework | Support In Place | Tape | Core in C++ | First Release Date |
|-----------|------------------|------|-------------|--------------------|
| Autograd  |                  | No   | No          | Mar 5, 2015        |
| Chainer   |                  | No   | No          | Jun 5, 2015        |
| Pytorch   | Yes              | No   | Yes         | Aug 31, 2016       |
| Dynet     | Yes              | Yes  | Yes         | Oct 12, 2016       |

### Autograd
[Backward code](https://github.com/HIPS/autograd/blob/442205dfefe407beffb33550846434baa90c4de7/autograd/core.py#L8-L40). In the forward pass, a graph of VJPNode is constructed.
```python
# User API
def make_grad(fun, x):
    start_node = VJPNode.new_root()
    end_value, end_node =  trace(start_node, fun, x)
    return backward_pass(g, end_node), end_value

# trace the forward pass by creating VJPNodes
def trace(start_node, fun, x):
    with trace_stack.new_trace() as t:
        start_box = new_box(x, t, start_node)
        end_box = fun(start_box)
        return end_box._value, end_box._node

def backward_pass(g, end_node):
    outgrads = {end_node : (g, False)}
    for node in toposort(end_node):
        outgrad = outgrads.pop(node)
        ingrads = node.vjp(outgrad[0])
        for parent, ingrad in zip(node.parents, ingrads):
            outgrads[parent] = add_outgrads(outgrads.get(parent), ingrad)
    return outgrad[0]

# Every VJPNode corresponds to a op_grad
class VJPNode(Node):
    __slots__ = ['parents', 'vjp']
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.parents = parents
        vjpmaker = primitive_vjps[fun]
        self.vjp = vjpmaker(parent_argnums, value, args, kwargs)
```
### Chainer
Example Code
```python
# (1) Function Set definition, creates FunctionNode
model = FunctionSet(
    l1=F.Linear(784, 100),
    l2=F.Linear(100, 100),
    l3=F.Linear(100, 10)).to_gpu()

# (2) Optimizer Setup
opt = optimizers.SGD()
opt.setup(model)

# (3) Forward computation
def forward(x, t):
    h1 = F.relu(model.l1(x))
    h2 = F.relu(model.l2(h1))
    y = model.l3(h2)
    return F.softmax_cross_entropy(y, t)

# (4) Training loop
for epoch in xrange(n_epoch):
    for i in xrange(0, N, b_size):
        x = Variable(to_gpu(...))
        t = Variable(to_gpu(...))
        opt.zero_grads()
        loss = forward(x, t)
        loss.backward()
        opt.update()
```
In `forward(x, t)`, a graph of [`VariableNode`](https://github.com/chainer/chainer/blob/master/chainer/variable.py#L110) and [`FunctionNode`](https://github.com/chainer/chainer/blob/a69103a4aa59d5b318f39b01dbcb858d465b89cf/chainer/function_node.py#L19) is constructed. Every output's `VariableNode.creator` is pointed to the `FunctionNode`.
```python
class FunctionNode(object):
    ...
    def apply(self, inputs):
        outputs = self.forward(inputs)
        ret = tuple([variable.Variable(y, requires_grad=requires_grad)
                     for y in outputs])
        # Topological ordering
        self.rank = max([x.rank for x in inputs]) if input_vars else 0
        # Add backward edges
        for y in ret:
            y.creator_node = self
        self.inputs = tuple([x.node for x in input_vars])
        self.outputs = tuple([y.node for y in ret])

        return ret
```
`loss.backward()` will calculate the accumulated gradient of all variables. All the backward of `FunctionNode`s will be called based on the topological order.
```python
class VariableNode(object):
    ...
    def backward(self, retain_grad, loss_scale):
        if self.creator_node is None:
            return

        cand_funcs = []
        seen_set = set()
        grads = {}

        # Initialize error by 1, if this is a loss variable
        if self.data.size == 1 and self._grad_var is None:
            self.grad = numpy.ones_like(self.data)
        grads[self._node] = self._grad_var

        def add_cand(cand):
            if cand not in seen_set:
                # Negate since heapq is min-heap. This is a global variable
                heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
                seen_set.add(cand)

        add_cand(self.creator_node)

        while cand_funcs:
            _, _, func = heapq.heappop(cand_funcs)
            gxs = func.backward_accumulate(func.inputs, func.outputs, func.outputs.grad)

            for x, gx in enumerate(gxs):
                if x in grads:
                    grads[x] += gx
                else:
                    grads[x] = gx

                if x.creator_node is not None:
                    add_cand(x.creator_node)
```

### PyTorch
Example Code
```python
x = Variable(torch.ones(5, 5))
y = Variable(torch.ones(5, 5) * 4)
z = x ** 2 + x * 2 + x * y + y
z.backward(torch.ones(5, 5))
```
The trace is done by `Variable.creator` and `Function.previous_functions`.
```python
class Variable(object):
    def __init__(self, tensor, creator=None, requires_grad=True):
        if creator is None:
            creator = Leaf(self, requires_grad)
        self.data = tensor
        self.creator = creator
        self._grad = None

    def backward(self, gradient=None):
        if gradient is None:
            if self.data.numel() != 1:
                raise RuntimeError('backward should be called only on a scalar (i.e. 1-element tensor) or with gradient w.r.t. the variable')
            gradient = self.data.new(1).fill_(1)
        self._execution_engine.run_backward(self, gradient)

class Function(obejct):
    # ...
    def _do_forward(self, *input):
        unpacked_input = tuple(arg.data for arg in input)
        raw_output = self.forward(*unpacked_input)

        # mark output.creator = self for backward trace
        output = tuple(Variable(tensor, self) for tensor in raw_output)

        self.previous_functions = [(arg.creator, id(arg)) for arg in input]
        self.output_ids = {id(var): i for i, var in enumerate(output)}
        return output

    def _do_backward(self, grad_output):
        return self.backwaerd(grad_output)
```
The [backward](https://github.com/pytorch/pytorch/blob/v0.1.1/torch/autograd/engine.py) is similar to Autograd.

### DyNet
Example code
```python
model = dy.model()
W_p = model.add_parameters((20, 100))
b_p = model.add_parameters(20)
E = model.add_lookup_parameters((20000, 50))
for epoch in range(num_epochs):
    for in_words, out_label in training_data:
        dy.renew_cg() # init tape
        W = dy.parameter(W_p)
        b = dy.parameter(b_p)
        score_sym = dy.softmax(W*dy.concatenate([E[in_words[0]],E[in_words[1]]])+b)
        loss_sym = dy.pickneglogsoftmax(score_sym, out_label)
        loss_val = loss_sym.value()
        loss_sym.backward()
```
[forward](https://github.com/clab/dynet/blob/740a9626a13a2732544de142e256ad0d0a166658/dynet/exec.cc#L84-L158), [backward](https://github.com/clab/dynet/blob/740a9626a13a2732544de142e256ad0d0a166658/dynet/exec.cc#L166-L284). The trace is done by creating a tape of expressions in every iteration. Backward is done by traverse the tape in the reverse order.
```c++
void SimpleExecutionEngine::backward(VariableIndex from_where, bool full) {
  ...  
  for (int i = num_nodes - 1; i >= 0; --i) {
    // each node corresponds to an op
    node->backward(xs, node_fx, node_dEdfx, ai, node_dEdxai);
  }
  ...
}
```



[1] [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/pdf/1502.05767.pdf)
