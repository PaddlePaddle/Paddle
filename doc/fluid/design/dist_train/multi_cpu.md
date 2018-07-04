# Design Doc: Execute the Program with Multi CPU

## Abstract

This Design Doc propose an approach to make the user-defined Op graph
running with multi-CPU, we will use an auto transpiler to convert the user-defined
Op graph to a multi-CPU Op graph, and run `ParallelDo` Op to run the graph.

## Transpiler

<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/single-thread@3x.png" width="300">

After converted:

<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/multi-threads@3x.png" width="1000">

## Implement

- `Multi-CPU Transpiler` will convert the graph to a multi-CPU graph
  which would be executed with multi-threads.
- `BlockingCounter` will `Init/Decrement` an atomic counter, and Blocking `Wait`
  for the atomic counter become `0`:
  ```cpp
  BlockingCounter bc(thread_count);
  for (int i = 0; i < thread_count; ++i) {
    thread_pool->Start([&bc] {bc.DecrementCount(); })
  }
  bc.Wait();
  ```
- `ParallelDo` Operator
  - Initialize a thread pool which is a Singleton.
  - Use a block id as the input, and create run the specify Block on independent scope
    with multi-threads.
  - Initialize a `BlockingCounter` instance and wait until all threads are done.
- `Split` Operator will split the Input Tensor into a TensorArray.
- `Merge` merge all the gradients which calculated in different threads
  with `mean/sum/max/min...` method, and then run the Optimizer Op to optimize `W`.

## TODO

- Improve the optimizer stage with multi-threads, since we could
  assign the parameters to the different threads and execute
  optimizer with multi-threads.
