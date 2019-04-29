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

#include "paddle/fluid/operators/math/concat_and_split.h"
#include <vector>

namespace paddle {
namespace operators {
namespace math {

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const std::vector<framework::Tensor>& input, int axis,
                  framework::Tensor* output, framework::Tensor* ins_info) {
    // TODO(zcd): Add input data validity checking
    int num = input.size();

    int rows = 1;
    auto dim_0 = input[0].dims();
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int out_rows = rows, out_cols = 0;

    std::vector<int64_t> input_cols(input.size());
    for (int i = 0; i < num; ++i) {
      int t_cols = input[i].numel() / rows;
      out_cols += t_cols;
      input_cols[i] = t_cols;
    }
    auto cpu_place = boost::get<platform::CPUPlace>(context.GetPlace());

    // computation
    auto output_data = output->data<T>();
    int col_idx = 0;
    for (int j = 0; j < num; ++j) {
      int col_len = input_cols[j];
      auto input_data = input[j].data<T>();
      for (int k = 0; k < out_rows; ++k) {
        memory::Copy(cpu_place, output_data + k * out_cols + col_idx, cpu_place,
                     input_data + k * col_len, sizeof(T) * col_len);
      }
      col_idx += col_len;
    }
  }
};

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class SplitFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input,
                  const std::vector<const framework::Tensor*>& ref_inputs,
                  const int axis, std::vector<framework::Tensor*>* outputs,
                  framework::Tensor* outs_info) {
    // TODO(zcd): Add input data validity checking
    size_t num = outputs->size();

    int input_rows = 1;
    auto dim_0 = ref_inputs[0]->dims();
    for (int i = 0; i < axis; ++i) {
      input_rows *= dim_0[i];
    }

    int input_cols = 0;

    std::vector<int64_t> output_cols(outputs->size());
    for (size_t i = 0; i < num; ++i) {
      int t_cols = ref_inputs[i]->numel() / input_rows;
      input_cols += t_cols;
      output_cols[i] = t_cols;
    }
    auto cpu_place = boost::get<platform::CPUPlace>(context.GetPlace());

    // computation
    for (int k = 0; k < input_rows; ++k) {
      const T* src_ptr = input.data<T>() + k * input_cols;
      int col_idx = 0;
      for (size_t j = 0; j < num; ++j) {
        int col_len = output_cols[j];
        auto* out_tensor = outputs->at(j);
        if (out_tensor != nullptr) {
          T* dst_ptr = out_tensor->data<T>() + k * col_len;
          memory::Copy(cpu_place, dst_ptr, cpu_place, src_ptr + col_idx,
                       sizeof(T) * col_len);
        }
        col_idx += col_len;
      }
    }
  }
};
#define DEFINE_FUNCTOR(type)                                      \
  template class ConcatFunctor<platform::CPUDeviceContext, type>; \
  template class SplitFunctor<platform::CPUDeviceContext, type>;

FOR_ALL_TYPES(DEFINE_FUNCTOR);

}  // namespace math
}  // namespace operators
}  // namespace paddle
