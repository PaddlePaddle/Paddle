#include "paddle/operators/random_op.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {
  
template<typename T>
class GaussianRandomOpKernel<platform::GPUPlace, T> : public framework::OpKernel {
public:
  void Compute(const framework::KernelContext& context) const override {
    auto mean = context.op_.GetAttr<T>("mean");
    auto std = context.op_.GetAttr<T>("std");
    auto* output = context.Output(0)->GetMutable<framework::Tensor>();
    T* r = output->mutable_data<T>(context.GetPlace());
    auto ctx = static_cast<const platform::GPUDeviceContext*>
      (context.device_context_);
    // generator need to modify context 
    auto g = const_cast<platform::GPUDeviceContext*>(ctx)->RandGenerator();
    curandGenerateNormal(g, r, framework::product(output->dims()), mean, std);

  }
};
  
}  // namespace operators
}  // namespace paddle
  

typedef paddle::operators::GaussianRandomOpKernel<paddle::platform::GPUPlace, float>
  RandomOpKernel_GPU_float;
REGISTER_OP_GPU_KERNEL(random, RandomOpKernel_GPU_float);