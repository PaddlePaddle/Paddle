// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/matrix_reduce.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"

namespace phi {
namespace funcs {

template <typename T>
class MatrixReduceSumFunctor<T, GPUContext> {
 public:
  void operator()(const GPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* out) {
    // For example: in's dim = [5, 3, 2, 7, 3] ; out's dim = [3, 1, 7, 3]
    // out_reduce_dim should be [0, 2]
    const std::vector<int> in_dims = phi::vectorize<int>(in.dims());
    auto in_size = in_dims.size();
    const std::vector<int> out_dims = phi::vectorize<int>(out->dims());
    auto out_size = out_dims.size();

    std::vector<int> out_bst_dims(in_size);

    std::fill(out_bst_dims.data(), out_bst_dims.data() + in_size - out_size, 1);
    std::copy(out_dims.data(),
              out_dims.data() + out_size,
              out_bst_dims.data() + in_size - out_size);
    out->Resize(phi::make_ddim(out_bst_dims));

    std::vector<int> out_reduce_dims;
    for (size_t idx = 0; idx <= in_size - 3; idx++) {
      if (in_dims[idx] != 1 && out_bst_dims[idx] == 1) {
        out_reduce_dims.push_back(idx);
      }
    }
    ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
        dev_ctx, in, out, kps::IdentityFunctor<T>(), out_reduce_dims);
  }
};

template class MatrixReduceSumFunctor<float, GPUContext>;
template class MatrixReduceSumFunctor<double, GPUContext>;

}  // namespace funcs
}  // namespace phi
