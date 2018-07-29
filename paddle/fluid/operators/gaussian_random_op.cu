/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/transform.h"

using paddle::platform::float16;

namespace paddle {
namespace operators {

template <typename T>
struct GaussianGenerator {
  T mean_, std_;
  unsigned int seed_;

  __host__ __device__ GaussianGenerator(T mean, T std, int seed)
      : mean_(mean), std_(std), seed_(seed) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::normal_distribution<T> dist(mean_, std_);
    rng.discard(n);
    return dist(rng);
  }
};

template <typename T, typename V>
struct CastFunctor {
  HOSTDEVICE V operator()(const T& a) { return static_cast<V>(a); }
};

template <typename T>
class GPUGaussianRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* tensor = context.Output<framework::Tensor>("Out");
    T* data = tensor->mutable_data<T>(context.GetPlace());
    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    if (seed == 0) {
      std::random_device rd;
      seed = rd();
    }
    T mean = static_cast<T>(context.Attr<float>("mean"));
    T std = static_cast<T>(context.Attr<float>("std"));
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    int64_t size = tensor->numel();

    if (std::type_index(typeid(T)) == std::type_index(typeid(float16))) {
      framework::Tensor master_copy_tensor;
      master_copy_tensor.Resize(tensor->dims());
      float* master_copy_tensor_data =
          master_copy_tensor.mutable_data<float>(context.GetPlace());
      thrust::transform(
          index_sequence_begin, index_sequence_begin + size,
          thrust::device_ptr<float>(master_copy_tensor_data),
          GaussianGenerator<float>(static_cast<float>(mean),
                                   static_cast<float>(std), seed));
      platform::Transform<platform::CUDADeviceContext> trans;
      auto* in_begin = master_copy_tensor.data<float>();
      auto* in_end = in_begin + master_copy_tensor.numel();
      auto* out_begin = tensor->mutable_data<T>(context.GetPlace());
      trans(context.template device_context<platform::CUDADeviceContext>(),
            in_begin, in_end, out_begin, CastFunctor<float, T>());
    } else {
      thrust::transform(index_sequence_begin, index_sequence_begin + size,
                        thrust::device_ptr<T>(data),
                        GaussianGenerator<T>(mean, std, seed));
    }

    if (VLOG_IS_ON(5)) {
      framework::Tensor cpu_tensor;
      framework::TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
      auto& dev_ctx =
          *platform::DeviceContextPool::Instance().Get(context.GetPlace());
      dev_ctx.Wait();
      auto x = framework::EigenVector<T>::Flatten(cpu_tensor);
      VLOG(5) << "The gaussian output " << x;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    gaussian_random, paddle::operators::GPUGaussianRandomKernel<float>,
    paddle::operators::GPUGaussianRandomKernel<double>,
    paddle::operators::GPUGaussianRandomKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(gaussian_random_batch_size_like,
                        paddle::operators::GPUGaussianRandomKernel<float>,
                        paddle::operators::GPUGaussianRandomKernel<double>);
