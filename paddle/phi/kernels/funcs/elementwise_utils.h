/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

namespace funcs {

using DDim = phi::DDim;

/*
 * Out = X ⊙ Y
 * If Y's shape does not match X' shape, they will be reshaped.
 * For example:
 * 1. shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
 *    pre=2, n=3*4, post=5
 *    x.shape(2, 12, 5) * y.shape(1, 12, 1).broadcast(2, 12, 5)
 * 2. shape(X) = (2, 3, 4, 5), shape(Y) = (4,5)
 *    pre=2*3, n=4*5, post=1
 *    x.shape(6, 20, 1) * y.shape(1, 20, 1).broadcast(6, 20, 1)
 *
 * New parameter: *is_run_common_broadcast* is a flag to record whether to run
 * common broadcast code.
 */
inline void GetMidDims(const DDim &x_dims,
                       const DDim &y_dims,
                       const int axis,
                       int *pre,
                       int *n,
                       int *post,
                       int *is_run_common_broadcast) {
  *pre = 1;
  *n = 1;
  *post = 1;
  *is_run_common_broadcast = 0;
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    if (x_dims[i + axis] != y_dims[i]) {
      PADDLE_ENFORCE_EQ(y_dims[i] == 1 || x_dims[i + axis] == 1,
                        true,
                        phi::errors::InvalidArgument(
                            "Broadcast dimension mismatch. Operands "
                            "could not be broadcast together with the shape of "
                            "X = [%s] and the shape of Y = [%s]. Received [%d] "
                            "in X is not equal to [%d] in Y.",
                            x_dims,
                            y_dims,
                            x_dims[i + axis],
                            y_dims[i]));
      *is_run_common_broadcast = 1;
      return;
    }
    (*n) *= y_dims[i];
  }
  for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
    (*post) *= x_dims[i];
  }
}

inline DDim TrimTrailingSingularDims(const DDim &dims) {
  // Remove trailing dimensions of size 1 for y
  auto actual_dims_size = dims.size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if (dims[actual_dims_size - 1] != 1) break;
  }
  if (actual_dims_size == dims.size()) return dims;
  std::vector<int> trim_dims;
  trim_dims.resize(actual_dims_size);
  for (int i = 0; i < actual_dims_size; ++i) {
    trim_dims[i] = dims[i];
  }
  if (trim_dims.size() == 0) {
    return DDim(phi::make_dim());
  }
  DDim actual_dims = phi::make_ddim(trim_dims);
  return actual_dims;
}

inline int GetElementwiseIndex(const int *x_dims_array,
                               const int max_dim,
                               const int *index_array) {
  int index_ = 0;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] > 1) {
      index_ = index_ * x_dims_array[i] + index_array[i];
    }
  }
  return index_;
}

inline void UpdateElementwiseIndexArray(const int *out_dims_array,
                                        const int max_dim,
                                        int *index_array) {
  for (int i = max_dim - 1; i >= 0; --i) {
    ++index_array[i];
    if (index_array[i] >= out_dims_array[i]) {
      index_array[i] -= out_dims_array[i];
    } else {
      break;
    }
  }
}

}  // namespace funcs
}  // namespace phi
