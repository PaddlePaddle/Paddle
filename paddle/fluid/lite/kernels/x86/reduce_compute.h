// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op_function.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

struct SumFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->sum(dim);
  }
};

#define HANDLE_DIM(NDIM, RDIM)                                            \
  if (ndim == NDIM && rdim == RDIM) {                                     \
    paddle::operators::ReduceFunctor<platform::CPUDeviceContext, T, NDIM, \
                                     RDIM, SumFunctor>(                   \
        platform::CPUDeviceContext(), input->raw_tensor(),                \
        &output->raw_tensor(), dims, keep_dim);                           \
  }

template <typename T>
class ReduceSumCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ReduceParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::ReduceParam>();
    // auto& context = context_->As<X86Context>();
    bool reduce_all = param.reduce_all;
    auto* input = param.x;
    auto* output = param.output;
    param.output->mutable_data<T>();

    auto dims = param.dim;
    bool keep_dim = param.keep_dim;
    if (reduce_all) {
      // Flatten and reduce 1-D tensor
      auto x = paddle::operators::EigenVector<T>::Flatten(input->raw_tensor());
      auto out = paddle::operators::EigenScalar<T>::From(output->raw_tensor());
      auto& place = *platform::CPUDeviceContext().eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({{0}});
      SumFunctor functor;
      functor(place, &x, &out, reduce_dim);
    } else {
      int ndim = input->dims().size();
      int rdim = dims.size();
      HANDLE_DIM(4, 3);
      HANDLE_DIM(4, 2);
      HANDLE_DIM(4, 1);
      HANDLE_DIM(3, 2);
      HANDLE_DIM(3, 1);
      HANDLE_DIM(2, 1);
      HANDLE_DIM(1, 1);
    }
  }

  virtual ~ReduceSumCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
