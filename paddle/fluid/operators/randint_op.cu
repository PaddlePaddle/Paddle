// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/uniform_random_op.h"

namespace paddle {
namespace operators {

template <typename T>
struct UniformGenerator {
  T low_, high_;
  __host__ __device__ UniformGenerator(T low, T high)
      : low_(low), high_(high) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(0);
    thrust::uniform_int_distribution<T> dist(low_, high_);
    rng.discard(n);
    T out = dist(rng);
    return out;
  }
};

// Use std::uniform_int_distribution and thrust::uniform_int_distribution(thrust
// is a std library in CUDA) to
// implement randint.
template <typename T>
class GPURandintKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
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

    auto* out = context.Output<framework::LoDTensor>("Out");
    if (!new_shape.empty()) out->Resize(framework::make_ddim(new_shape));
    T* data = out->mutable_data<T>(context.GetPlace());
    T low = static_cast<T>(context.Attr<int>("low"));
    T high = static_cast<T>(context.Attr<int>("high")) - 1;

    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    int64_t size = out->numel();
    thrust::transform(index_sequence_begin, index_sequence_begin + size,
                      thrust::device_ptr<T>(data),
                      UniformGenerator<T>(low, high));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(randint, ops::GPURandintKernel<int>,
                        ops::GPURandintKernel<int64_t>,
                        ops::GPURandintKernel<float>,
                        ops::GPURandintKernel<double>);
