//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/pull_sparse_op.h"

namespace paddle {
namespace operators {

template <typename T>
class PullSparseV2CPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PullSparseFunctor<T>(ctx, &fea_keys_, &pull_result_ptr_,
                         &pull_sparse_status_);
  }

 protected:
  std::vector<uint64_t> fea_keys_;
  std::vector<T*> pull_result_ptr_;
  std::vector<::std::future<int32_t>> pull_sparse_status_;
};

template <typename T>
class PushSparseV2CPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PushSparseFunctor<T>(ctx, &push_keys_, &push_values_, &fea_labels_);
  }

 protected:
  std::vector<uint64_t> push_keys_;
  std::vector<std::vector<T>> push_values_;
  std::vector<T> fea_labels_;
};
}  // namespace operators
}  // namespace paddle
