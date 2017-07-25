#include "paddle/operators/random_op.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
bool Gaussian(platform::CUDADeviceContext &ctx, framework::Tensor* output,
              const int size, const T& mean, const T& std, const T& seed) {
  auto g = RandGenerator(seed);
  return curandGenerateNormal(g, output, size, mean, std);
}

} // operators
} // paddle


typedef paddle::operators::RandomOpKernel<paddle::platform::GPUPlace, float>
  RandomOpKernel_GPU_float;
REGISTER_OP_GPU_KERNEL(random, RandomOpKernel_GPU_float);