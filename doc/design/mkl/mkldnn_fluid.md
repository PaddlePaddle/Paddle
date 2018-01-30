# Design Doc: Add MKLDNN Kernel in Fluid Operator

## Principles

First of all, we should follow some basical principles like:
1.  [How to write a new operator](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/howto/dev/new_op_en.md). We are trying to add a new kind of kernel into operators, so basically we should follow this doc.
2.  [Supporting new Device/Library](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/support_new_device.md). Since MKLDNN is a new library to fluid, we should add `MKLDNNDeviceContext` and maybe `mkldnn_helper.h`, just like [cudnn_helper.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/platform/cudnn_helper.h).
3.  [Switch Kernel](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/switch_kernel.md). Another important point is that we should ensure the data synchronization between different kernel types, which is this [topic](https://github.com/PaddlePaddle/Paddle/issues/6549). So basically we should override `GetExpectedKernelType` and `trans` functions to support switching kernels.
4.  [The Keys of Operator Kernel Type](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/operator_kernel_type.md). Kernel Type is a pivotal conception which can record the `Place`, `Library`, `DataType` and `Layout`.

## Sulution

In general, there are four parts we should follow to run a MKL-DNN primitive.
-  Create a primitive descriptor that describe this operator
-  Create a primitive itself by primitive descriptor and the engine
-  Create all memory buffers that primitive needed
-  Launch a stream to execute the primitive created
More details can refer to [here](http://01org.github.io/mkl-dnn).

It's better to avoid reinitialization of primitives and memory handles in the first three stages in every iteration. \
So we plan to create a map to record all the `primitive` and `memory`, which should not take too much memories as discussed [here](https://github.com/PaddlePaddle/Paddle/issues/6822).

It's assumed that following three conditions should be satisfied.
1. there is a unique key for each operator instance. May be the actual name of `Output Tensor`.
2. the `Input Tensor` inside `Compute` function is the one after converted.
3. we can get the phase(eg. `is_test`) inside `Compute` function, otherwise we need to expose this attribue to user.

### Compute
The algorithm of `Compute` would be described as follow, let's take conv like an example.

```c++

  PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()), "It must use CPUPlace.");
  PADDLE_ENFORCE(platform::is_mkldnn_library(ctx.GetLibrary()), "It must use MKLDNN Library.");

  auto& dev_ctx = ctx.template device_context<platform::MKLDNNDeviceContext>();

  // find primitive by unique key from mkldnn context
  // the op_key should be a unique name of this op instance
  auto& p = dev_ctx.findPrimitive(op_key + "_fwd");

  // assuming the input tensor inside this compute function is the one after converted
  // this point should be guarantee by another mechanism
  auto& i = dev_ctx.findMemory(op_key + "_input");
  
  if (p == nullptr || i == nullptr || inputSizeChanged(p, i))  {
    auto fwd_primitive_desc = createPrimitiveDesc(ctx);
    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");
    shared_ptr<mkldnn::memory> in(new mkldnn::memory(fwd_primitive_desc->src_primitive_desc(), input->data<T>()));
    shared_ptr<mkldnn::memory> wgt(new mkldnn::memory(fwd_primitive_desc->weights_primitive_desc(), filter->data<T>()));
    shared_ptr<mkldnn::memory> out(new mkldnn::memory(fwd_primitive_desc->dst_primitive_desc(), output->mutable_data<T>(ctx.GetPlace())));
    shared_ptr<mkldnn::conv_fwd> fwd_primitive(new mkldnn::conv_fwd(*fwd_primitive_desc, *in, *wgt, *out));

    dev_ctx.addMemory(op_key+"_input", in);
    dev_ctx.addMemory(op_key+"_output", out);
    dev_ctx.addMemory(op_key+"_filer", wgt);
    dev_ctx.addPrimitive(op_key+"_fwd", fwd_primitive);
    dev_ctx.addPrimitiveDesc(op_key+"_fwd_PD", fwd_primitive_desc);
  }

  p = dev_ctx.findPrimitive(op_key + "_fwd");

  PADDLE_ENFORCE(p, "Should have forward Primitive");
  PADDLE_ENFORCE(dev_ctx.findMemory(op_unique_key+"_input"), "Should have input memory");
  PADDLE_ENFORCE(dev_ctx.findMemory(op_unique_key+"_output"), "Should have output memory");
  PADDLE_ENFORCE(dev_ctx.findMemory(op_unique_key+"_filter"), "Should have filter memory");
  PADDLE_ENFORCE(dev_ctx.findPrimitiveDesc(op_unique_key+"_fwd_PD"), "Should have forward PrimitiveDesc");
  dev_ctx.submit(p);
  dev_ctx.execute();  // the convert primitive should have already contained.

```

The `createPrimitiveDesc` returns the primitive descripotor of this operator, would be like this:
```c++
  auto* input = ctx.Input<Tensor>("Input");
  auto* filter = ctx.Input<Tensor>("Filter");
  auto* output = ctx.Output<Tensor>("Output");
  std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
  std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
  std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
  int groups = ctx.Attr<int>("groups");
  algorithm algo = static_cast<algorithm>(ctx.Attr<int>("convolution_algorithm_option"));
  prop_kind pk = ctx.Attr<bool>("is_test") ? prop_kind::forward_inference : prop_kind::forward_training;
    
  auto fwd_desc = mkldnn::conv_fwd::desc(/* all the setting above*/);
  shared_ptr<mkldnn::conv_fwd::primitive_desc> fwd_primitive_desc(new mkldnn::conv_fwd::primitive_desc(fwd_desc, ctx.getEngine()));

  return fwd_primitive_desc;
  }
```

### MKLDNNDeviceContext
`MKLDNNDeviceContext`, which is very straightforward, should contain some base information like: `stream`, `engine` and the map needed.


### mkldnn_helper
Some functions would be put in `paddle/platform/mkldnn_helper.h`.
- create MKLDNN memories
- create MKLDNN primitives
- error check function
- etc


### Kernel Switch
We should `reorder` the different Layout from other device or to other device. `GetExpectedKernelType` and `trans` functions can help us to implement it.

`GetExpectedKernelType` should get the context, and this operator can return the best `KernelType`. 
`trans` would be like this:

```c++
void trans(inputs, ctx) override {
  if (NoNeedTrans()) {
    return;
  }
  // find reorder primitive by op_key from context
  auto& dev_ctx = ctx.template device_context<platform::MKLDNNDeviceContext>();
  auto& p = dev_ctx.findPrimitive(op_key + "_reorder_input");
  auto& i = dev_ctx.findMemory(op_key + "_src_input");

  if (p == nullptr || i == nullptr || changeSized(i, input)) {
    auto prim = createPrimitiveDesc(ctx);
    auto src = createMemory(memoryDesc(input->dims(), actual_layout), input->data);
    auto newbuffer = paddle::memory::Alloc(ctx.GetPlace(), input->size_in_bytes());
    auto dst = createMemory(p->expected_desc(), newbuffer->data);
    auto reorder_primitive(new mkldnn::reorder(src, dst));

    dev_ctx.addMemory(op_key+"_src_input", src);
    dev_ctx.addMemory(op_key+"_input", dst);
    dev_ctx.addPrimitive(op_key+"_reorder_input", reorder_primitive);
  }

  p = dev_ctx.findPrimitive(op_key + "_reorder_input");
  PADDLE_ENFORCE(p, "Should have Reorder Primitive");
  dev_ctx.submit(p);
  if (! this->isMKLDNNKernel()) {
    // execute immediately only if this is not mkldnn kernel function.
    // otherwise, it can be executed with the operator primitive in Compute
    dev_ctx.stream();
  }
  // after submit, the input tensor in ExecutionContext should be changed as the converted one
  // there should be another mechanism to ensure this
}
```

### Unit Test
All the functions should be tested corresponding.
TBD
