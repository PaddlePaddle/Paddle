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

#include "paddle/phi/kernels/index_sample_kernel.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
namespace phi {
template <typename T, typename Context, typename IndexT = int>
void IndexSampleInner(const Context &context,
                      const DenseTensor &input,
                      const DenseTensor &index,
                      DenseTensor *output) {
  auto input_dims = input.dims();
  auto index_dims = index.dims();

  int batch_size = input_dims[0];
  auto value_length = input_dims[1];
  auto index_length = index_dims[1];
  int index_ids_num = index.numel();

  output->Resize({batch_size, index_length});
  context.template Alloc<T>(output);
  auto input_tensor = EigenVector<T>::Flatten(input);
  auto index_tensor = EigenVector<IndexT>::Flatten(index);
  auto output_tensor = EigenVector<T>::Flatten(*output);

  for (int i = 0; i < index_ids_num; i++) {
    int b = floor(i / index_length);
    PADDLE_ENFORCE_GE(
        index_tensor(i),
        0,
        errors::InvalidArgument(
            "Variable value (index) of OP(index_sample) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            value_length,
            index_tensor(i)));
    PADDLE_ENFORCE_LT(
        index_tensor(i),
        value_length,
        errors::InvalidArgument(
            "Variable value (index) of OP(index_sample) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            value_length,
            index_tensor(i)));

    int v_i = b * value_length + static_cast<int>(index_tensor(i));
    T v = input_tensor(v_i);
    VLOG(4) << "Index Sample: batch = " << b << " index = " << v_i
            << " value = " << v;
    output_tensor(i) = v;
  }
  auto output_dims = output->dims();
  output->Resize(output_dims);
}

template <typename T, typename Context>
void IndexSampleKernel(const Context &ctx,
                       const DenseTensor &x,
                       const DenseTensor &index,
                       DenseTensor *out) {
  ctx.template Alloc<T>(out);
  auto index_type = index.dtype();
  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      errors::InvalidArgument(
          "Input(Index) holds the wrong type, it holds %s, but "
          "desires to be %s or %s",
          paddle::framework::DataTypeToString(
              paddle::framework::TransToProtoVarType(index_type)),
          paddle::framework::DataTypeToString(
              paddle::framework::TransToProtoVarType(DataType::INT32)),
          paddle::framework::DataTypeToString(
              paddle::framework::TransToProtoVarType((DataType::INT64)))));
  if (index_type == DataType::INT32) {
    IndexSampleInner<T, Context, int>(ctx, x, index, out);
  } else if (index_type == DataType::INT64) {
    IndexSampleInner<T, Context, int64_t>(ctx, x, index, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_sample,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexSampleKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {}
