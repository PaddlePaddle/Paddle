/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#define EIGEN_USE_GPU
#include <memory>
#include <typeindex>
#include <typeinfo>

#include <curand_kernel.h>

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/dropout_op.h"

namespace paddle {
namespace operators {

// http://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes
// seperate curand_init as a single kernel for performance.
template <typename AttrType>
__global__ void CurandInitKernel(curandState* state, AttrType seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(static_cast<unsigned long long>(seed), idx, 0, &state[idx]);
}

template <typename T, typename AttrType>
__global__ void UniformRandomkernel(curandState* state, AttrType dropout_prob,
                                    int64_t n, T* dst) {
  int idx = threadIdx.x + blockIdx.x * 64;
  int64_t count = 0;
  /* Copy state to local memory for efficiency */
  T rand_val;
  curandState local_state = state[idx];
  for (int64_t i = 0; i < n; ++i) {
    if (std::type_index(typeid(T)).hash_code() == typeid(float).hash_code()) {
      rand_val = curand_uniform(&local_state);
      dst[i] = rand_val < dropout_prob;
    }
  }
  /*Copy state back to global memory */
  state[idx] = local_state;
}

template <typename Place, typename T, typename AttrType>
class GPUDropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    y->mutable_data<T>(context.GetPlace());
    AttrType dropout_prob = context.Attr<AttrType>("dropout_prob");

    auto X = EigenMatrix<T>::Reshape(*x, 1);
    auto Y = EigenMatrix<T>::Reshape(*y, 1);

    auto& place = *context.template device_context<Place>().eigen_device();
    if (!context.Attr<bool>("is_test")) {
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<T>(context.GetPlace());
      int64_t size = framework::product(mask->dims());

      std::random_device rnd;
      int seed =
          context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();

      // FIXME(dzhwinter): create state without hard core code.
      curandState* dev_states;
      std::unique_ptr<curandState> dev_states(
          memory::Alloc<platform::CUDAPlace>(context.GetPlace(),
                                             sizeof(curandState) * 64 * 64),
          memory::PlainDeleter<curandState, platform::CUDAPlace>(place_));
      CurandInitKernel<<<64, 64>>>(dev_states.get());
      UniformRandomkernel<<<64, 64>>>(dev_states.get(), dropout_prob, size,
                                      mask_data);

      auto M = EigenMatrix<T>::Reshape(*mask, 1);
      Y.device(place) = X * M;
    } else {
      Y.device(place) = X * (1.0f - dropout_prob);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    dropout,
    ops::GPUDropoutKernel<paddle::platform::CUDADeviceContext, float, float>);
REGISTER_OP_CUDA_KERNEL(
    dropout_grad,
    ops::DropoutGradKernel<paddle::platform::CUDADeviceContext, float>);
