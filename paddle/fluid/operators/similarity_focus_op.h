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
#include <cstring>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class SimilarityFocusKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    phi::DenseTensor* out = context.Output<phi::DenseTensor>("Out");
    const phi::DenseTensor* x = context.Input<phi::DenseTensor>("X");
    T* out_data = out->mutable_data<T>(context.GetPlace());
    const T* x_data = x->data<T>();

    int axis = context.Attr<int>("axis");
    std::vector<int> indexes = context.Attr<std::vector<int>>("indexes");

    int64_t batch_size = x->dims()[0];
    int64_t dim[4];
    for (int i = 1; i <= 3; ++i) {
      dim[i] = x->dims()[i];
    }

    PADDLE_ENFORCE_GT(
        indexes.size(),
        0,
        platform::errors::InvalidArgument("The size of Attr(indexes) must be "
                                          "greater than 0, but received %d.",
                                          indexes.size()));

    for (size_t i = 0; i < indexes.size(); i++) {
      PADDLE_ENFORCE_GT(
          dim[axis],
          indexes[i],
          platform::errors::InvalidArgument(
              "Each value of Attr(indexes) must be less than X.dim[axis], "
              "but indexes[%d] received %d.",
              i,
              indexes[i]));
    }

    int64_t array_size = 1;
    for (int i = 1; i <= 3; ++i) {
      if (i != axis) {
        array_size *= dim[i];
      }
    }

    std::vector<std::pair<T, int64_t>> array(array_size);

    bool (*cmp)(std::pair<T, int64_t>, std::pair<T, int64_t>) =
        [](std::pair<T, int64_t> x, std::pair<T, int64_t> y) {
          return x.first > y.first;
        };

    int64_t (*compute_index)(int64_t*, int, int, int, int) =
        [](int64_t* dim, int d1, int d2, int d3, int d4) {
          return d1 * dim[1] * dim[2] * dim[3] + d2 * dim[2] * dim[3] +
                 d3 * dim[3] + d4;
        };

    PADDLE_ENFORCE_GT(
        axis,
        0,
        platform::errors::InvalidArgument(
            "The value of Attr(axis) must be 1 or 2 or 3, but received %d.",
            axis));
    PADDLE_ENFORCE_LT(
        axis,
        4,
        platform::errors::InvalidArgument(
            "The value of Attr(axis) must be 1 or 2 or 3, but received %d.",
            axis));
    memset(out_data, 0, sizeof(T) * batch_size * dim[1] * dim[2] * dim[3]);
    for (int i = 0; i < batch_size; ++i) {
      for (auto index : indexes) {
        if (axis == 1) {
          for (int j = 0; j < dim[2]; ++j) {
            for (int k = 0; k < dim[3]; ++k) {
              array[j * dim[3] + k] = std::make_pair(
                  x_data[compute_index(dim, i, index, j, k)], j * dim[3] + k);
            }
          }

          std::sort(array.begin(), array.end(), cmp);
          int tag_num = 0;
          std::vector<bool> tag2(dim[2]), tag3(dim[3]);
          for (auto x : array) {
            int idx2 = x.second / dim[3];
            int idx3 = x.second % dim[3];
            if (tag2[idx2] || tag3[idx3]) {
              continue;
            }
            tag_num++;
            tag2[idx2] = true;
            tag3[idx3] = true;
            for (int j = 0; j < dim[1]; ++j) {
              out_data[compute_index(dim, i, j, idx2, idx3)] = 1;
            }
            if (tag_num == std::min(dim[2], dim[3])) {
              break;
            }
          }
        } else if (axis == 2) {
          for (int j = 0; j < dim[1]; ++j) {
            for (int k = 0; k < dim[3]; ++k) {
              array[j * dim[3] + k] = std::make_pair(
                  x_data[compute_index(dim, i, j, index, k)], j * dim[3] + k);
            }
          }

          std::sort(array.begin(), array.end(), cmp);
          int tag_num = 0;
          std::vector<bool> tag1(dim[1]), tag3(dim[3]);
          for (auto x : array) {
            int idx1 = x.second / dim[3];
            int idx3 = x.second % dim[3];
            if (tag1[idx1] || tag3[idx3]) {
              continue;
            }
            tag_num++;
            tag1[idx1] = true;
            tag3[idx3] = true;
            for (int j = 0; j < dim[2]; ++j) {
              out_data[compute_index(dim, i, idx1, j, idx3)] = 1;
            }
            if (tag_num == std::min(dim[1], dim[3])) {
              break;
            }
          }
        } else if (axis == 3) {
          for (int j = 0; j < dim[1]; ++j) {
            for (int k = 0; k < dim[2]; ++k) {
              array[j * dim[2] + k] = std::make_pair(
                  x_data[compute_index(dim, i, j, k, index)], j * dim[2] + k);
            }
          }

          std::sort(array.begin(), array.end(), cmp);
          int tag_num = 0;
          std::vector<bool> tag1(dim[1]), tag2(dim[2]);
          for (auto x : array) {
            int idx1 = x.second / dim[2];
            int idx2 = x.second % dim[2];
            if (tag1[idx1] || tag2[idx2]) {
              continue;
            }
            tag_num++;
            tag1[idx1] = true;
            tag2[idx2] = true;
            for (int j = 0; j < dim[3]; ++j) {
              out_data[compute_index(dim, i, idx1, idx2, j)] = 1;
            }
            if (tag_num == std::min(dim[1], dim[2])) {
              break;
            }
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
