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
    int axis = ctx.Attr<int>("axis");

    auto in_dims = input->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;

    const T* in_data = input->data<T>();
    T* out_data = output->mutable_data<T>(ctx.GetPlace());
    int64_t* ids_data = indices->mutable_data<int64_t>(ctx.GetPlace());

    int64_t groups = input->numel() / in_dims[axis];
    int64_t stride = (axis == in_dims.size() - 1)
                         ? 1
                         : framework::product(framework::slice_ddim(
                               in_dims, axis + 1, in_dims.size()));

    for (int64_t i = 0; i < groups; ++i) {
      int64_t idx = i;
      std::vector<int64_t> shape_vec(in_dims.size(), 0);
      for (int64_t dim = in_dims.size() - 1; dim >= 0; --dim) {
        if (dim != axis) {
          shape_vec[dim] = idx % in_dims[dim];
          idx /= in_dims[dim];
        }
      }

      int64_t start_index = shape_vec[0];
      for (int64_t dim = 0; dim < in_dims.size() - 1; ++dim) {
        start_index = start_index * in_dims[dim + 1] + shape_vec[dim + 1];
      }

      std::vector<int64_t> org_index_vec(in_dims[axis], start_index);
      for (int64_t j = 1; j < in_dims[axis]; ++j) {
        org_index_vec[j] += j * stride;
      }

      std::sort(org_index_vec.begin(), org_index_vec.end(),
                [in_data](const int64_t v1, const int64_t v2) {
                  return in_data[v1] < in_data[v2];
                });

      for (size_t j = 0; j < org_index_vec.size(); ++j) {
        int64_t index = start_index + j * stride;
        out_data[index] = in_data[org_index_vec[j]];
        ids_data[index] = (org_index_vec[j] - start_index) / stride;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
