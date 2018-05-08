//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/math/blas.h"

#include <utility>
namespace paddle {
namespace operators {
namespace math {
MatDescriptor GetMatDim(const framework::DDim& dim, int num_flatten_cols,
                        bool trans) {
  MatDescriptor retv;
  if (num_flatten_cols > 1) {
    auto flatten_dim = framework::flatten_to_2d(dim, num_flatten_cols);
    retv.height_ = flatten_dim[0];
    retv.width_ = flatten_dim[1];
  } else {
    if (dim.size() == 1) {
      retv.height_ = 1;
      retv.width_ = dim[0];
    } else if (dim.size() == 2) {
      retv.height_ = dim[0];
      retv.width_ = dim[1];
    } else {
      if (dim.size() == 3) {
        retv.batch_size_ = dim[0];
        retv.height_ = dim[1];
        retv.width_ = dim[2];
      } else {
        auto dim_vec = framework::vectorize(dim);
        retv.batch_size_ = 1;
        for (size_t i = 0; i < dim_vec.size() - 2; ++i) {
          retv.batch_size_ *= dim_vec[i];
          retv.height_ = dim_vec[dim_vec.size() - 2];
          retv.width_ = dim_vec[dim_vec.size() - 1];
        }
      }
      retv.stride_ = retv.height_ * retv.width_;
    }
  }
  if (trans) {
    std::swap(retv.width_, retv.height_);
  }
  retv.trans_ = trans;
  return retv;
}
}  // namespace math
}  // namespace operators
}  // namespace paddle
