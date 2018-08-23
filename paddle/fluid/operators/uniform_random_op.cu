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

namespace paddle {
namespace operators {

template <typename T>
struct UniformGenerator {
  T min_, max_;
  unsigned int seed_;

  __host__ __device__ UniformGenerator(T min, T max, int seed)
      : min_(min), max_(max), seed_(seed) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(n);
    return dist(rng);
  }
};

// It seems that Eigen::Tensor::random in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename T>
class GPUUniformRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    framework::Tensor* tensor = nullptr;
    auto out_var = context.OutputVar("Out");
    if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
    } else if (out_var->IsType<framework::SelectedRows>()) {
      auto shape = context.Attr<std::vector<int>>("shape");
      tensor = out_var->GetMutable<framework::SelectedRows>()->mutable_value();
      tensor->Resize(framework::make_ddim(shape));
    } else {
      PADDLE_THROW(
          "uniform_random_op's output only"
          "supports SelectedRows and Tensor");
    }
    T* data = tensor->mutable_data<T>(context.GetPlace());
    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    if (seed == 0) {
      std::random_device rd;
      seed = rd();
    }
    T min = static_cast<T>(context.Attr<float>("min"));
    T max = static_cast<T>(context.Attr<float>("max"));
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    int64_t size = tensor->numel();
    thrust::transform(index_sequence_begin, index_sequence_begin + size,
                      thrust::device_ptr<T>(data),
                      UniformGenerator<T>(min, max, seed));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(uniform_random,
                        paddle::operators::GPUUniformRandomKernel<float>,
                        paddle::operators::GPUUniformRandomKernel<double>);
REGISTER_OP_CUDA_KERNEL(uniform_random_batch_size_like,
                        paddle::operators::GPUUniformRandomKernel<float>,
                        paddle::operators::GPUUniformRandomKernel<double>);
