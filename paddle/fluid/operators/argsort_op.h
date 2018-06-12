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
#include <algorithm>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ArgsortKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("X");
    auto* output = ctx.Output<framework::Tensor>("Out");
    auto* indices = ctx.Output<framework::Tensor>("Indices");
    int axis = static_cast<int>(ctx.Attr<int>("axis"));

    auto in_dims = input->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;

    const T* in_data = input->data<T>();
    T* out_data = output->mutable_data<T>(ctx.GetPlace());
    int64_t* idx_data = indices->mutable_data<int64_t>(ctx.GetPlace());

    int64_t part_dims_prod = input->numel() / in_dims[axis];
    for (int64_t i = 0; i < part_dims_prod; ++i) {
      int64_t idx = i;
      std::vector<int64_t> idx_vec(in_dims.size(), 0);
      for (int64_t dim = in_dims.size() - 1; dim >= 0; --dim) {
        if (dim != axis) {
          idx_vec[dim] = idx % in_dims[dim];
          idx /= in_dims[dim];
        }
      }
      std::vector<std::pair<T, int64_t>> in_vec;
      std::vector<int64_t> org_index_vec(in_dims[axis], 0);
      for (int64_t j = 0; j < in_dims[axis]; ++j) {
        idx_vec[axis] = j;
        int64_t index = idx_vec[0];
        for (int64_t dim = 0; dim < in_dims.size() - 1; ++dim) {
          index = index * in_dims[dim + 1] + idx_vec[dim + 1];
        }
        in_vec.push_back(std::pair<T, int64_t>(in_data[index], j));
        org_index_vec[j] = index;
      }

      std::sort(
          in_vec.begin(), in_vec.end(),
          [](const std::pair<T, int64_t>& v1, const std::pair<T, int64_t>& v2) {
            return v1.first < v2.first;
          });

      for (size_t j = 0; j < org_index_vec.size(); ++j) {
        int64_t index = org_index_vec[j];
        out_data[index] = in_vec[j].first;
        idx_data[index] = in_vec[j].second;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
