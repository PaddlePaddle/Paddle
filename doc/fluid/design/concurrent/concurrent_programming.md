# Design Doc: Concurrent Programming with Fluid

With PaddlePaddle Fluid, users describe a program other than a model.  The program is a [`ProgramDesc`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto) protobuf message. TensorFlow/MxNet/Caffe2 applications generate protobuf messages too, but their protobuf messages represent the model, a graph of operators, but not the program that trains/uses the model.   

Many know that when we program TensorFlow, we can specify the device on which each operator runs.  This allows us to create a concurrent/parallel AI application.   An interesting questions is **how does a `ProgramDesc` represents a concurrent program?**  

The answer relies on the fact that a `ProgramDesc` is similar to an abstract syntax tree (AST) that describes a program.  So users just program a concurrent program that they do with any concurrent programming language, e.g., [Go](https://golang.org).

## An Analogy

The following table compares concepts in Fluid and Go

<table>
<thead>
<tr>
<th></th>
<th>Go</th>
<th>Fluid</th>
</tr>
</thead>
<tbody>
<tr>
<td>user-defined functions </td>
<td>
<a href="https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid">layers</a></td>
<td></td>
</tr>
<tr>
<td>control-flow and built-in functions </td>
<td>
<a href="https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators">intrinsics/operators</a></td>
<td></td>
</tr>
<tr>
<td>goroutines, channels </td>
<td>
<a href="https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/framework/thread_pool.h">class ThreadPool</a></td>
<td></td>
</tr>
<tr>
<td>runtime </td>
<td>
<a href="https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/executor.h">class Executor</a></td>
<td></td>
</tr>
</tbody>
</table>


## An Example Concurrent Program

To review all above concepts in an example, let us take a simple program and writes its distributed version.

Suppose that we want to parallelize a naive Fluid program (written in Go and calling Fluid's Go binding) that multiplies two tensors.

```go
import "fluid"

func paddlepaddle() {
  X = fluid.read(...)
  W = fluid.Tensor(...)
  Y = fluid.mult(X, W)
}
```

Please be aware that the Fluid's Go binding provides the default `main` function, which calls the `paddlepaddle` function, which, in this case, is defined in above program and creates the following `ProgramDesc` message.

```protobuf
message ProgramDesc {
  block[0] = Block {
    vars = [X, W, Y],
    ops = [
      read(output = X)
      assign(input = ..., output = W)
      mult(input = {X, W}, output = Y)
    ],
  }
}
```

Then, the default `main` function calls `fluid.run()`, which creates an instance of the [`class Executor`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/executor.h) and calls `Executor.Run(block[0])`, where `block[0]` is the first and only block defined in above `ProgramDesc` message.

The default `main` function is defined as follows:

```go
func main() {
  paddlepaddle()
  fluid.run()
}
```

## The Concurrent Version

By parallelizing the above program, we could support very big tensor X by splitting into small pieces {x_1, x_2, ...} and sent each piece to worker process/node for parallel multiplication.

In this case, we can write a transpiler that takes a `ProgramDesc` message that represents the above example program and outputs two `ProgramDesc` messages, one for running on the master process/node, and the other one for worker processes/nodes.

### The Master Program

The master program could look like the following:

```protobuf
message ProgramDesc {
  block[0] = Block {
    vars = [X, L, Y],
    ops = [
      read(output = X)
      kube_get_workers_addrs(output = L)
      Y = tensor_array(len(L))
      parallel_for(input = X, output = Y,
                   attrs = {L, block_id(1)}) # referring to block 1
    ]
  }

  block[1] = Block {
    parent = 0,
    vars = [x, y, index],
    ops = [
      slice(input = [X, index], output = x) # index is initialized by parallel_for
      send(input = x, attrs = L[index])
      recv(outputs = y, attrs = L[index])
      assign(input = y, output = Y[index])
    ]
  }
}
```

The equivalent Fluid program (calling the Go binding) is:

```go
func main() {  //// block 0
  X = fluid.read(...)
  L = fluid.k8s.get_worker_addrs()
  Y = fluid.tensor_array(len(L))
  fluid.parallel_for(X, L,
                     func(index int) {  //// block 1
                       x = X[index]
                       fluid.send(L[index], x)
                       y = fluid.recv(L[index])
                       Y[index] = y
                     })
}
```

An explanation of the above program:

- `fluid.k8s` is a package that provides access to Kubernetes API.  
- `fluid.k8s.get_worker_addrs` returns the list of IP and ports of all pods of the current job except for the current one (the master pod).  
- `fluid.tensor_array` creates a [tensor array](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/lod_tensor_array.h).  `fluid.parallel_for` creates a `ParallelFor` intrinsic, which, when executed,

  1. creates `len(L)` scopes, each for the concurrent running of the sub-block (block 1 in this case), and initializes a variable named "index" in the scope to an integer value in the range `[0, len(L)-1]`, and
  2. creates `len(L)` threads by calling into the `ThreadPool` singleton, each thread  
     1. creates an Executor instance, and
     2. calls `Executor.Run(block)`, where `block` is block 1 as explained above.
1. Please be aware that block 1 is a sub-block of block 0, so ops in block 1 could refer to variables defined in block 0.

### The Worker Program

The worker program looks like

```go
func main() {
  W = Tensor(...)
  x = fluid.listen_and_do(
        fluid.k8s.self_addr(),
        func(input Tensor) {
          output = fluid.mult(input, W)
        })
}
```

where

- `fluid.listen_and_do` creates a `ListenAndDo` intrinsic, which, when executed,
  1. listens on the current pod's IP address, as returned by `fliud.k8s.self_addr()`,
  2. once a connection is established,
     1. creates a scope of two parameters, "input" and "output",
     2. reads a [Fluid variable](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/variable.h) and saves it into "input",
     3. creates an Executor instance and calls `Executor.Run(block)`, where the block is generated by running the lambda specified as the second parameter of `fluid.listen_and_do`.

## Summarization

From the above example, we see that:

1. Fluid enables the imperative programming paradigm by:
   1. letting users describe a program, but not a model (a sequence of layers, or a graph of operators), and
   2. call the `fluid.run` function that runs the program implicitly.
1. The program is described as a `ProgramDesc` protobuf message.
2. Function `Executor.Run` takes a block, instead of a `ProgramDesc`, as its parameter.
3. `fluid.run` calls `Executor.Run` to run the first block in the `ProgramDesc` message.
4. `Executor.Run`'s implementation is extremely simple -- it doesn't plan the execution nor create threads; instead, it runs on the current thread and execute intrinsics/operators' `Run` method sequentially as they appear in the `Block.ops` array.
5. Intrinsics/operators' `Run` method might create threads.  For example, the `ListenAndDo` operator creates a thread to handle each incoming request.
6. Threads are not necessarily OS thread; instead, they could be [green threads](https://en.wikipedia.org/wiki/Green_threads) managed by ThreadPool.  Multiple green threads might run on the same OS thread.  An example green threads is Go's [goroutines](https://tour.golang.org/concurrency/1).
