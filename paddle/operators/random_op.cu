#include "paddle/operators/random_op.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using paddle::platform::GPUPlace;
template<GPUPlace, typename T, typename Generator>
bool Gaussian(Generator g, T* output, const int size, const T& mean, const T& std) {
  return curandGenerateNormal(g, output, size, mean, std);
}

} // operators
} // paddle


typedef paddle::operators::RandomOpKernel<paddle::platform::GPUPlace, float>
  RandomOpKernel_GPU_float;
REGISTER_OP_GPU_KERNEL(random_op, RandomOpKernel_GPU_float);