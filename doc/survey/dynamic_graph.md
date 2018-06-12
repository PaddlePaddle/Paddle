# Automatic Differentiation with the Tape

## Automatic Differentiation

A key challenge in the field of deep learning is to automatically derive the backward pass from the forward pass described algorithmically by researchers.  Such a derivation, or a transformation of the forward pass program, has been long studied before the recent prosperity of deep learning in the field known as [automatic differentiation](https://arxiv.org/pdf/1502.05767.pdf).

## The Tape

Given the forward pass program (usually in Python in practices), there are two strategies to derive the backward pass:

1. from the forward pass program itself, or
1. from the execution trace of the forward pass program, which is often known as the *tape*.

This article surveys systems that follow the latter strategy.

## Dynamic Network

When we train a deep learning model, the tape changes every iteration as the input data change, so we have to re-derive the backward pass every iteration.  This is known as *dynamic network*.

Deep learning systems that utilize the idea of dynamic network gained their popularities in recent years.  This article surveys two representative systems: [PyTorch](https://pytorch.org/) and [DyNet](https://dynet.readthedocs.io/en/latest/).

## An Overview

Both frameworks record a ‘tape’ of the computation and interpreting (or run-time compiling) a transformation of the tape played back in reverse. This tape is a different kind of entity than the original program.[[link]](http://www.bcl.hamilton.ie/~barak/papers/toplas-reverse.pdf)

Consider the following code feedforward model.

```python
x = Variable(randn(20, 1)))
label = Variable(randint(1))
W_1, W_2 = Variable(randn(20, 20)), Variable(randn(10, 20))
h = matmul(W_1, x)
pred = matmul(W_2, x)
loss = softmax(pred, label)
loss.backward()
```

### 1) Dynet uses List to encode the Tape

During the forward execution, a list of operators, in this case `matmul`, `matmul` and `softmax`, are recorded in the tape, along with the necessary information needed to do the backward such as pointers to the inputs and outputs. Then the tape is played in reverse order at `loss.backward()`.

<details> 
<summary></summary>
digraph g {
    graph [
        rankdir = "LR"
    ];
    node [
        fontsize = "16"
        shape = "ellipse"
    ];
    edge [];
    "node0" [
        label = "<f0> type: matmul | <f1> input: W_1, x | <f2> output: h"
        shape = "record"
    ];
    "node1" [
        label = "<f0> type: matmul | <f1> input: W_2, h | <f2> output: pred"
        shape = "record"
    ];
    "node2" [
        label = "<f0> type: softmax | <f1> input: pred, label | <f2> output: loss"
        shape = "record"
    ];
    "node0":f0 -> "node1":f0 [];
    "node1":f0 -> "node2":f0 [];
}
</details>

![Alt text](https://g.gravizo.com/svg?digraph%20g%20{%20graph%20[%20rankdir%20=%20%22LR%22%20];%20node%20[%20fontsize%20=%20%2216%22%20shape%20=%20%22ellipse%22%20];%20edge%20[];%20%22node0%22%20[%20label%20=%20%22%3Cf0%3E%20type:%20matmul%20|%20%3Cf1%3E%20input:%20W_1,%20x%20|%20%3Cf2%3E%20output:%20h%22%20shape%20=%20%22record%22%20];%20%22node1%22%20[%20label%20=%20%22%3Cf0%3E%20type:%20matmul%20|%20%3Cf1%3E%20input:%20W_2,%20h%20|%20%3Cf2%3E%20output:%20pred%22%20shape%20=%20%22record%22%20];%20%22node2%22%20[%20label%20=%20%22%3Cf0%3E%20type:%20softmax%20|%20%3Cf1%3E%20input:%20pred,%20label%20|%20%3Cf2%3E%20output:%20loss%22%20shape%20=%20%22record%22%20];%20%22node0%22:f0%20-%3E%20%22node1%22:f0%20[%20id%20=%200%20];%20%22node1%22:f0%20-%3E%20%22node2%22:f0%20[%20id%20=%201%20];%20})

### 2) Pytorch uses Node Graph to encode the Tape

The graph is composed of `Variable`s and `Function`s. During the forward execution, a `Variable` records its creator function, e.g. `h.creator = matmul`. And a Function records its inputs' previous/dependent functions `prev_func` through `creator`, e.g. `matmul.prev_func = matmul1`. At `loss.backward()`, a topological sort is performed on all `prev_func`s. Then the grad op is performed by the sorted order.

<details> 
<summary></summary>
digraph g {
    graph [
        rankdir = "LR"
    ];
    
    subgraph function {
        node [
            fontsize = "16"
            style = filled
            shape = "record"
        ];
        "matmul0" [ label = "<f0> type: matmul | prev_func: None" ];
        "matmul1" [ label = "<f0> type: matmul | prev_func: matmul" ];
        "softmax" [ label = "<f0> type: softmax | prev_func: matmul" ];
    }
    
    subgraph variable {
        node [
            fontsize = "16"
            shape = "Mrecord"
            style = filled
            fillcolor = white
        ];
        "x" [ label = "<f0> x | <f1> creator: None" ];
        "label" [ label = "<f0> label | <f1> creator: None" ];
        "W_1" [ label = "<f0> W_1 | <f1> creator: None" ];
        "W_2" [ label = "<f0> W_2 | <f1> creator: None" ];
        "h" [ label = "<f0> h | <f1> creator: None" ];
        "pred" [ label = "<f0> pred | <f1> creator: matmul" ];
        "loss" [ label = "<f0> loss | <f1> creator: softmax" ];
    }
    
    subgraph data_flow {
        "x":f0 -> "matmul0":f0;
        "W_1":f0 -> "matmul0":f0;
        "matmul0":f0 -> "h":f0;
    
        "h":f0 -> "matmul1":f0;
        "W_2":f0 -> "matmul1":f0;
        "matmul1":f0 -> "pred":f0;
    
        "pred":f0 -> "softmax":f0;
        "label":f0 -> "softmax":f0;
        "softmax":f0 -> "loss":f0;
    }

    subgraph prev_func {
        edge [color="red", arrowsize="0.6", penwidth="1", constraint=false];
        "matmul1":f1 -> "matmul0":f0;
        "softmax":f1 -> "matmul1":f0;
        label = "prev_func";
    }
}
</details>

![Alt text](https://g.gravizo.com/svg?digraph%20g%20{%20graph%20[%20rankdir%20=%20%22LR%22%20];%20subgraph%20function%20{%20node%20[%20fontsize%20=%20%2216%22%20style%20=%20filled%20shape%20=%20%22record%22%20];%20%22matmul0%22%20[%20label%20=%20%22%3Cf0%3E%20type:%20matmul%20|%20prev_func:%20None%22%20];%20%22matmul1%22%20[%20label%20=%20%22%3Cf0%3E%20type:%20matmul%20|%20prev_func:%20matmul%22%20];%20%22softmax%22%20[%20label%20=%20%22%3Cf0%3E%20type:%20softmax%20|%20prev_func:%20matmul%22%20];%20}%20subgraph%20variable%20{%20node%20[%20fontsize%20=%20%2216%22%20shape%20=%20%22Mrecord%22%20style%20=%20filled%20fillcolor%20=%20white%20];%20%22x%22%20[%20label%20=%20%22%3Cf0%3E%20x%20|%20%3Cf1%3E%20creator:%20None%22%20];%20%22label%22%20[%20label%20=%20%22%3Cf0%3E%20label%20|%20%3Cf1%3E%20creator:%20None%22%20];%20%22W_1%22%20[%20label%20=%20%22%3Cf0%3E%20W_1%20|%20%3Cf1%3E%20creator:%20None%22%20];%20%22W_2%22%20[%20label%20=%20%22%3Cf0%3E%20W_2%20|%20%3Cf1%3E%20creator:%20None%22%20];%20%22h%22%20[%20label%20=%20%22%3Cf0%3E%20h%20|%20%3Cf1%3E%20creator:%20None%22%20];%20%22pred%22%20[%20label%20=%20%22%3Cf0%3E%20pred%20|%20%3Cf1%3E%20creator:%20matmul%22%20];%20%22loss%22%20[%20label%20=%20%22%3Cf0%3E%20loss%20|%20%3Cf1%3E%20creator:%20softmax%22%20];%20}%20subgraph%20data_flow%20{%20%22x%22:f0%20-%3E%20%22matmul0%22:f0;%20%22W_1%22:f0%20-%3E%20%22matmul0%22:f0;%20%22matmul0%22:f0%20-%3E%20%22h%22:f0;%20%22h%22:f0%20-%3E%20%22matmul1%22:f0;%20%22W_2%22:f0%20-%3E%20%22matmul1%22:f0;%20%22matmul1%22:f0%20-%3E%20%22pred%22:f0;%20%22pred%22:f0%20-%3E%20%22softmax%22:f0;%20%22label%22:f0%20-%3E%20%22softmax%22:f0;%20%22softmax%22:f0%20-%3E%20%22loss%22:f0;%20}%20subgraph%20prev_func%20{%20edge%20[color=%22red%22,%20arrowsize=%220.6%22,%20penwidth=%221%22,%20constraint=false];%20%22matmul1%22:f1%20-%3E%20%22matmul0%22:f0;%20%22softmax%22:f1%20-%3E%20%22matmul1%22:f0;%20label%20=%20%22prev_func%22;%20}%20})

Chainer and Autograd uses the similar techniques to record the forward pass. For details please refer to the appendix.

## Design choices

### 1) Dynet's List vs Pytorch's Node Graph

What's good about List:
1. It avoids a topological sort. One only needs to traverse the list of operators in reverse and calling the corresponding backward operator.
1. It promises effient data parallelism implementations. One could count the time of usage of a certain variable during the construction list. Then in the play back, one knows the calculation of a variable has completed. This enables communication and computation overlapping.

What's good about Node Graph:
1. More flexibility. PyTorch users can mix and match independent graphs however they like, in whatever threads they like (without explicit synchronization). An added benefit of structuring graphs this way is that when a portion of the graph becomes dead, it is automatically freed. [[2]](https://openreview.net/pdf?id=BJJsrmfCZ) Consider the following example, Pytorch only does backward on SmallNet while Dynet does both BigNet and SmallNet.
```python
result = BigNet(data)
loss = SmallNet(data)
loss.backward()
```

### 2) Dynet's Lazy evaluation vs Pytorch's Immediate evaluation

Dynet builds the list in a symbolic matter. Consider the following example
```python
for epoch in range(num_epochs):
    for in_words, out_label in training_data:
        dy.renew_cg()
        W = dy.parameter(W_p)
        b = dy.parameter(b_p)
        score_sym = dy.softmax(W*dy.concatenate([E[in_words[0]],E[in_words[1]]])+b)
        loss_sym = dy.pickneglogsoftmax(score_sym, out_label)
        loss_val = loss_sym.value()
        loss_sym.backward()
```
The computation of `lookup`, `concat`, `matmul` and `softmax` didn't happen until the call of `loss_sym.value()`. This defered execution is useful because it allows some graph-like optimization possible, e.g. kernel fusion.

Pytorch chooses immediate evaluation. It avoids ever materializing a "forward graph"/"tape" (no need to explicitly call `dy.renew_cg()` to reset the list), recording only what is necessary to differentiate the computation, i.e. `creator` and `prev_func`.


## What can fluid learn from them?

Please refer to `paddle/contrib/dynamic/`.

# Appendix

### Overview

| Framework | Has Tape | Core in C++ | First Release Date |
|-----------|----------|-------------|--------------------|
| Autograd  | No       | No          | Mar 5, 2015        |
| Chainer   | No       | No          | Jun 5, 2015        |
| Pytorch   | No       | Yes         | Aug 31, 2016       |
| Dynet     | Yes      | Yes         | Oct 12, 2016       |

### Source Code
#### Autograd
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
#### Chainer
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

#### PyTorch
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

#### DyNet
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
