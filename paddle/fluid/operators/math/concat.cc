/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/concat.h"

namespace paddle {
namespace operators {
namespace math {

/*
 * All tensors' dimension should be the same.
 */
template <typename T>
class ConcatFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  std::vector<framework::Tensor>& input, const int axis,
                  framework::Tensor* output) {
    // assume the the max size of input is less than 8 and see the performance
    // save origin dim
    int num = input.size();
    std::vector<paddle::framework::DDim> origin_dim(num);
    //    for (int j = 0; j < num; ++j) {
    //      origin_dim[j] = input[j].dims();
    //    }
    auto out_dim = output->dims();

    // get the matrix size
    int rows = 1;
    auto dim_0 = input[0].dims();
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int cols = input[0].numel() / rows;
    int out_rows = rows, out_cols = 0;
    bool sameShape = true;

    // reshape to matrix
    for (int i = 0; i < num; ++i) {
      int t_cols = input[i].numel() / rows;
      if (sameShape) {
        if (t_cols != cols) sameShape = false;
      }
      out_cols += t_cols;
      input[i].Resize({rows, t_cols});
    }
    output->Resize({out_rows, out_cols});
    auto& cpu_place = boost::get<platform::CPUPlace>(context.GetPlace());
    // computation
    for (int k = 0; k < rows; ++k) {
      // offset k * out_cols
      T* dst_ptr = output->data<T>() + k * out_cols;
      int col_idx = 0;
      for (int j = 0; j < num; ++j) {
        int col_len = input[j].dims()[1];
        const T* src_prt = input[j].data<T>() + k * col_len;
        memory::Copy(cpu_place, dst_ptr + col_idx, cpu_place, src_prt,
                     sizeof(T) * col_len);
        col_idx += col_len;
      }
    }

    // recover origin dim
    //    for (int j = 0; j < num; ++j) {
    //      input[j]->Resize(origin_dim[j]);
    //    }
    output->Resize(out_dim);
  }
};

template class ConcatFunctor<platform::CPUDeviceContext, int>;
template class ConcatFunctor<platform::CPUDeviceContext, int64_t>;
template class ConcatFunctor<platform::CPUDeviceContext, float>;
template class ConcatFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
