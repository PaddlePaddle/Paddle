# Design Doc: Execute the Program with Multi Thread

In PaddlePaddle, the user could declare **wich** operators will be
execute and **how much** threads will be run in Python code.

## Python Interface

We can specify the thread count with `parallel.do(thread_count=N)`,
the following PaddlePaddle program shows the usage of the Parallel operator:

```python
import paddle.v2.fluid as fluid
N=4
parallel = fluid.parallel()

x = minibatch([10, 20, 30, 40, 50])
y = var("y")
label = var(1)
w = var("w")
b = var("b")

fluid.split(x, N)

with parallel.do(thread_count=N):
    y_predict = fluid.fc(input=x, size=1)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(x=cost)

fluid.merge_grad_avg(x, N) # merge gradient with avg/sum/max/min/...
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)

with parallel.do(thread_count=N):
    sgd_optimizer.minimize(avg_cost)

```

## Operator Kernel

- We need a global threadpool, and initialize the threadpool when the
    Executor run the Parallel Op for the first time.
- The Op kernel will create an Executor instance, and send the blocks which number
    is `thread_count` into the threadpool and run the block for each thread, while
    all the blocks is done, the Op will exit, it's a sync process.

## Operators

- `Split` Op will split the Tensor into N tensors
- `MergeGradAvg` Op will merge the gradients with `avg` , we also need to
    implement other math such as `sum/max/min...`.
