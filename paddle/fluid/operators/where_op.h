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

#pragma once
#include <functional>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct WhereFunctor {
  WhereFunctor(const T& true_index, int true_num, const T& stride, int rank,
               int64_t* out)
      : true_index_(true_index),
        true_num_(true_num),
        stride_(stride),
        rank_(rank),
        out_ptr_(out) {}

  HOSTDEVICE void operator()(size_t idx) const {
    int index = true_index_[idx];
    for (int j = 0; j < rank_; j++) {
      out_ptr_[idx * rank_ + j] = index / stride_[j];
      index -= out_ptr_[idx * rank_ + j] * stride_[j];
    }
  }

  const T true_index_;
  int true_num_;
  const T stride_;
  int rank_;
  int64_t* out_ptr_;
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

using CPUDeviceContext = paddle::platform::CPUDeviceContext;

template <typename T>
class CPUWhereKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::Tensor>("Condition");
    auto* out = context.Output<framework::Tensor>("Out");

    const bool* cond_data = condition->data<bool>();
    auto numel = condition->numel();
    auto dims = condition->dims();
    const int rank = dims.size();

    std::vector<int> true_index;
    for (auto i = 0; i < numel; i++) {
      if (cond_data[i]) {
        true_index.push_back(i);
      }
    }
    auto true_num = true_index.size();

    out->Resize(framework::make_ddim({static_cast<int64_t>(true_num), rank}));
    auto out_ptr = out->mutable_data<T>(context.GetPlace());

    if (true_num == 0) {
      return;
    }

    std::vector<int> stride(rank);
    stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * dims[i + 1];
    }

    auto& dev_ctx = context.template device_context<CPUDeviceContext>();
    WhereFunctor<int*> functor(true_index.data(), true_num, stride.data(), rank,
                               out_ptr);
    platform::ForRange<CPUDeviceContext> for_range(dev_ctx, true_num);
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle
