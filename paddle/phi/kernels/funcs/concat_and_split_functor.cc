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

#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/common/enforce.h"

namespace phi::funcs {

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
struct ConcatFunctor<CPUContext, T> {
  void operator()(const CPUContext& context,
                  const std::vector<DenseTensor>& input,
                  int axis,
                  DenseTensor* output) {
    // TODO(zcd): Add input data validity checking
    size_t num = input.size();

    int64_t rows = 1;
    auto dim_0 = input[0].dims();
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int64_t out_rows = rows, out_cols = 0;

    PADDLE_ENFORCE_NE(
        rows,
        0,
        common::errors::InvalidArgument("The input size should not be 0."));

    std::vector<int64_t> input_cols(input.size());
    for (size_t i = 0; i < num; ++i) {
      int64_t t_cols = input[i].numel() / rows;
      out_cols += t_cols;
      input_cols[i] = t_cols;
    }
    auto cpu_place = context.GetPlace();

    // computation
    auto output_data = output->data<T>();
    int64_t col_idx = 0;
    for (size_t j = 0; j < num; ++j) {
      int64_t col_len = input_cols[j];
      auto input_data = input[j].data<T>();
      for (int64_t k = 0; k < out_rows; ++k) {
        memory_utils::Copy(cpu_place,
                           output_data + k * out_cols + col_idx,
                           cpu_place,
                           input_data + k * col_len,
                           sizeof(T) * col_len);
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
struct SplitFunctor<CPUContext, T> {
 public:
  void operator()(const CPUContext& context,
                  const DenseTensor& input,
                  const std::vector<const DenseTensor*>& ref_inputs,
                  int axis,
                  std::vector<DenseTensor*>* outputs) {
    // NOTE(zhiqiu): split a tensor of shape [0,3,4] at axis=1, result in 3
    // tensors of shape [0,1,4]
    if (input.numel() == 0) {
      return;
    }

    // TODO(zcd): Add input data validity checking
    size_t num = outputs->size();

    int64_t input_rows = 1;
    auto dim_0 = ref_inputs[0]->dims();
    for (int i = 0; i < axis; ++i) {
      input_rows *= dim_0[i];
    }

    int64_t input_cols = 0;

    std::vector<int64_t> output_cols(outputs->size());
    for (size_t i = 0; i < num; ++i) {
      int64_t t_cols = ref_inputs[i]->numel() / input_rows;
      input_cols += t_cols;
      output_cols[i] = t_cols;
    }
    auto cpu_place = context.GetPlace();

    // computation
    for (int64_t k = 0; k < input_rows; ++k) {
      const int64_t src_offset = k * input_cols;
      const T* src_ptr = input.data<T>() + src_offset;
      int64_t col_idx = 0;
      for (size_t j = 0; j < num; ++j) {
        int64_t col_len = output_cols[j];
        auto* out_tensor = outputs->at(j);
        if (out_tensor != nullptr) {
          const int64_t dst_offset = k * col_len;
          T* dst_ptr = out_tensor->data<T>() + dst_offset;
          memory_utils::Copy(cpu_place,
                             dst_ptr,
                             cpu_place,
                             src_ptr + col_idx,
                             sizeof(T) * col_len);
        }
        col_idx += col_len;
      }
    }
  }
};

#define DEFINE_FUNCTOR(type)                                 \
  template class PADDLE_API ConcatFunctor<CPUContext, type>; \
  template class PADDLE_API SplitFunctor<CPUContext, type>;

FOR_ALL_TYPES(DEFINE_FUNCTOR);

}  // namespace phi::funcs
