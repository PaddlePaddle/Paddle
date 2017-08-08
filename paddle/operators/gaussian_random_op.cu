#include <memory>
#include <random>
#include "paddle/platform/dynload/curand.h"
#include "paddle/platform/gpu_info.h"

#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class GaussianRandomKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    T mean = static_cast<T>(context.op_.GetAttr<T>("mean"));
    T std = static_cast<T>(context.op_.GetAttr<T>("std"));
    auto* tensor = context.Output<framework::Tensor>(0);
    T* data = tensor->mutable_data<T>(context.GetPlace());

    int seed = context.op_.GetAttr<int>("seed");
    if (seed == 0) {
      seed = std::random_device()();
    }
    curandGenerator_t g;
    PADDLE_ENFORCE(platform::dynload::curandCreateGenerator(
        &g, CURAND_RNG_PSEUDO_DEFAULT));
    PADDLE_ENFORCE(
        platform::dynload::curandSetPseudoRandomGeneratorSeed(g, seed));
    // auto g = const_cast<platform::GPUDeviceContext*>(ctx)->RandGenerator();
    curandGenerateNormal(g, data, framework::product(tensor->dims()), mean,
                         std);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(gaussian_random, ops::GaussianRandomKernel<float>);