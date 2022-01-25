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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/distribution_helper.h"
#include "paddle/fluid/operators/uniform_random_op.h"

DECLARE_bool(use_curand);

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

template <typename T>
struct UniformGeneratorOffset {
  T min_, max_;
  unsigned int seed_;
  T diag_val_;
  unsigned int diag_num_;
  unsigned int diag_step_;
  int offset_;
  __host__ __device__ UniformGeneratorOffset(T min, T max, int seed,
                                             int diag_num, int diag_step,
                                             T diag_val, int offset)
      : min_(min),
        max_(max),
        seed_(seed),
        diag_num_(diag_num),
        diag_step_(diag_step),
        diag_val_(diag_val),
        offset_(offset) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(n + offset_);
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

    if (out_var->IsType<pten::SelectedRows>()) {
      auto* selected_rows = out_var->GetMutable<pten::SelectedRows>();
      tensor = selected_rows->mutable_value();
      auto shape = context.Attr<std::vector<int64_t>>("shape");
      if (!new_shape.empty()) shape = new_shape;
      tensor->Resize(framework::make_ddim(shape));
      selected_rows->mutable_rows()->reserve(shape[0]);
    } else if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
      if (!new_shape.empty()) tensor->Resize(framework::make_ddim(new_shape));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Output(out) in uniform_random_op must be Tensor, "
          "SelectedRows. But got "
          "unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }
    auto& dev_cxt =
        context.template device_context<platform::CUDADeviceContext>();
    T* data = tensor->mutable_data<T>(dev_cxt.GetPlace());
    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    bool seed_flag = false;
    if (seed == 0) {
      std::random_device rd;
      seed = rd();
      seed_flag = true;
    }

    T min = static_cast<T>(context.Attr<float>("min"));
    T max = static_cast<T>(context.Attr<float>("max"));
    unsigned int diag_num =
        static_cast<unsigned int>(context.Attr<int>("diag_num"));
    unsigned int diag_step =
        static_cast<unsigned int>(context.Attr<int>("diag_step"));
    T diag_val = static_cast<T>(context.Attr<float>("diag_val"));
    thrust::counting_iterator<int64_t> index_sequence_begin(0);
    int64_t size = tensor->numel();
    int device_id = context.GetPlace().GetDeviceId();
    auto gen_cuda = framework::GetDefaultCUDAGenerator(device_id);
    if (gen_cuda->GetIsInitPy() && seed_flag) {
      if (FLAGS_use_curand) {
        using MT = typename details::MPTypeTrait<T>::Type;
        distribution::uniform_distribution<MT> dist;
        distribution::uniform_transform<MT> trans(min, max);
        distribution::distribution_and_transform<T>(dev_cxt, tensor, dist,
                                                    trans);
      } else {
        auto seed_offset = gen_cuda->IncrementOffset(1);
        int64_t gen_offset = size * seed_offset.second;
        thrust::transform(
            index_sequence_begin, index_sequence_begin + size,
            thrust::device_ptr<T>(data),
            UniformGeneratorOffset<T>(min, max, seed_offset.first, diag_num,
                                      diag_step, diag_val, gen_offset));
      }
    } else {
      thrust::transform(
          index_sequence_begin, index_sequence_begin + size,
          thrust::device_ptr<T>(data),
          UniformGenerator<T>(min, max, seed, diag_num, diag_step, diag_val));
    }
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
