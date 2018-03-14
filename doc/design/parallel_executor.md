# ParallelExecutor Design Doc

## Introduction

We introduce `ParallelExecutor` to run multi-GPU training in PaddlePaddle Fluid. It supports
1. keeping a copy of the parameters on each GPU
1. allreduce on a separate stream allowing computation and communication overlap

An example of switching single GPU training to multiple GPUs:
```python
cost = your_neural_network()
opt = fluid.optimizer.SGDOptimizer()
opt.minimize(avg_cost)

# change Executor -> ParallelExecutor
exe = fluid.ParallelExecutor(gpu_list=[0, 1])

for iter in xranges(iter_num):
    exe.run()
```

## Design

In the constructor, a list of parameter, whose gradients need to be allreduced, is given.

During the runtime, `ParallelExecutor` starts `#gpu` threads to run each `Executor`. For every
operator run on each GPU, it will automatically sync with different streams when necessary.

```c++
// if op's input is params' grad:
    // sync with allreduce stream
    // e.g. sgd should wait for allreduce to be finished
CallBack->BeforeOp(op);

op->Run(*local_scope, place_);

// if op's output is params' grad:
//     sync with computation stream
//     e.g. allreduce shoudl wait for fc_grad to be finished.
CallBack->AfterOp(op);
```

And the `Callback` object can be implemented as the following

```c++
struct AllReduceCallBack {
  void BeforeOp(framework::OperatorBase* op);
  void AfterOp(framework::OperatorBase* op);

  std::unordered_set<std::string> reduced_param_grad_names;
  std::unordered_set<std::string> param_grad_names_;

  platform::DeviceContext* computation_dev_ctx;    // computation device context
  platform::DeviceContext* communication_dev_ctx;  // communication device context

  framework::Scope* scope;
  platform::NCCL::Communicator* nccl_com;
};

AllReduceCallBack::BeforeOp(framework::OperatorBase* op) {
  if (op->Input() in reduced_param_grad_names) {
    communication_dev_ctx->Wait();
    reduced_param_grad_names.erase(op->Input())
  }
}

AllReduceCallBack::AfterOp(framework::OperatorBase* op) {
  if (op->Output() in param_grad_names) {
    computation_dev_ctx->Wait();
    reduced_param_grad_names.insert(op->Output());
    ncclAllreduce(scope, op->Output(), communication_dev_ctx);
  }
}
```
