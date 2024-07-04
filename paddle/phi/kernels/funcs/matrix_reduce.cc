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
#include "paddle/phi/kernels/funcs/reduce_functor.h"

namespace phi::funcs {

template <typename T>
class MatrixReduceSumFunctor<T, CPUContext> {
 public:
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* out) {
    // For example: in's dim = [5, 3, 2, 7, 3] ; out's dim = [3, 1, 7, 3]
    // out_reduce_dim should be [0, 2]
    const std::vector<int64_t> in_dims = common::vectorize<int64_t>(in.dims());
    auto in_size = in_dims.size();
    const std::vector<int64_t> out_dims =
        common::vectorize<int64_t>(out->dims());
    auto out_size = out_dims.size();

    std::vector<int64_t> out_bst_dims(in_size);

    std::fill(out_bst_dims.data(), out_bst_dims.data() + in_size - out_size, 1);
    std::copy(out_dims.data(),
              out_dims.data() + out_size,
              out_bst_dims.data() + in_size - out_size);
    out->Resize(common::make_ddim(out_bst_dims));

    std::vector<int64_t> out_reduce_dims;
    for (size_t idx = 0; idx <= in_size - 3; idx++) {
      if (in_dims[idx] != 1 && out_bst_dims[idx] == 1) {
        out_reduce_dims.push_back(static_cast<int>(idx));
      }
    }
    ReduceKernelImpl<CPUContext, T, T, phi::funcs::SumFunctor>(
        dev_ctx, in, out, out_reduce_dims, true, false);
  }
};

template class MatrixReduceSumFunctor<float, CPUContext>;
template class MatrixReduceSumFunctor<double, CPUContext>;
template class MatrixReduceSumFunctor<phi::dtype::complex<float>, CPUContext>;
template class MatrixReduceSumFunctor<phi::dtype::complex<double>, CPUContext>;

}  // namespace phi::funcs
