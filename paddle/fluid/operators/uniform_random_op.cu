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
#include "paddle/fluid/operators/uniform_random_op.h"
namespace paddle {
namespace operators {

template <typename T>
struct UniformGenerator {
  T min_, max_;
  unsigned int seed_;
  T diag_val_;
  unsigned int diag_num_;
  unsigned int diag_step_;
  __host__ __device__ UniformGenerator(T min, T max, int seed, int diag_num,
                                       int diag_step, T diag_val)
      : min_(min),
        max_(max),
        seed_(seed),
        diag_num_(diag_num),
        diag_step_(diag_step),
        diag_val_(diag_val) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(n);
    T out = dist(rng);
    unsigned int remainder = n % (diag_step_ + 1);
    if (remainder == 0 && diag_num_ > n / (diag_step_ + 1)) {
      out = diag_val_;
    }
    return out;
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
    std::vector<int64_t> new_shape;
    auto list_new_shape_tensor =
        context.MultiInput<framework::Tensor>("ShapeTensorList");
    if (list_new_shape_tensor.size() > 0 || context.HasInput("ShapeTensor")) {
      if (context.HasInput("ShapeTensor")) {
        auto* shape_tensor = context.Input<framework::Tensor>("ShapeTensor");
        new_shape = GetNewDataFromShapeTensor(shape_tensor);
      } else if (list_new_shape_tensor.size() > 0) {
        new_shape = GetNewDataFromShapeTensorList(list_new_shape_tensor);
      }
    }

    if (out_var->IsType<framework::SelectedRows>()) {
      auto* selected_rows = out_var->GetMutable<framework::SelectedRows>();
      tensor = selected_rows->mutable_value();
      auto shape = context.Attr<std::vector<int64_t>>("shape");
      if (!new_shape.empty()) shape = new_shape;
      tensor->Resize(framework::make_ddim(shape));
      selected_rows->mutable_rows()->reserve(shape[0]);
    } else if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
      if (!new_shape.empty()) tensor->Resize(framework::make_ddim(new_shape));
    } else {
      PADDLE_THROW(
          "uniform_random_op's output only"
          "supports SelectedRows and LoDTensor");
    }
    T* data = tensor->mutable_data<T>(context.GetPlace());
    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    if (seed == 0) {
      std::random_device rd;
      seed = rd();
    }
    T min = static_cast<T>(context.Attr<float>("min"));
    T max = static_cast<T>(context.Attr<float>("max"));
    unsigned int diag_num =
        static_cast<unsigned int>(context.Attr<int>("diag_num"));
    unsigned int diag_step =
        static_cast<unsigned int>(context.Attr<int>("diag_step"));
    T diag_val = static_cast<T>(context.Attr<float>("diag_val"));
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    int64_t size = tensor->numel();
    thrust::transform(
        index_sequence_begin, index_sequence_begin + size,
        thrust::device_ptr<T>(data),
        UniformGenerator<T>(min, max, seed, diag_num, diag_step, diag_val));
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
