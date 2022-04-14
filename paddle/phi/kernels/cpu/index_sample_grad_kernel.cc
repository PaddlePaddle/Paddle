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

#include "paddle/phi/kernels/index_sample_grad_kernel.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
namespace phi {
template <typename T, typename Context, typename IndexT = int>
void IndexSampleGradInner(const Context& context,
                          const DenseTensor& out_grad,
                          const DenseTensor& index,
                          DenseTensor* x_grad) {
  std::vector<T> out_grad_vec;
  std::vector<IndexT> index_vec;
  paddle::framework::TensorToVector(out_grad, context, &out_grad_vec);
  paddle::framework::TensorToVector(index, context, &index_vec);

  auto index_dims = index.dims();
  auto x_grad_dims = x_grad->dims();

  auto value_length = x_grad_dims[1];
  auto index_length = index_dims[1];
  int index_ids_num = index.numel();

  std::vector<T> x_grad_vec(x_grad->numel(), 0);

  for (int i = 0; i < index_ids_num; i++) {
    int b = floor(i / index_length);
    PADDLE_ENFORCE_GE(
        index_vec[i],
        0,
        errors::InvalidArgument(
            "Variable value (index) of OP(index_sample_grad) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            value_length,
            index_vec[i]));
    PADDLE_ENFORCE_LT(
        index_vec[i],
        value_length,
        errors::InvalidArgument(
            "Variable value (index) of OP(index_sample_grad) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            value_length,
            index_vec[i]));
    int v_i = b * value_length + static_cast<int>(index_vec[i]);
    x_grad_vec[v_i] += out_grad_vec[i];
  }
  context.template Alloc<T>(x_grad);
  paddle::framework::TensorFromVector(x_grad_vec, context, x_grad);
  x_grad->Resize(x_grad_dims);
}

template <typename T, typename Context>
void IndexSampleGradKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& index,
                           const DenseTensor& out_grad,
                           DenseTensor* x_grad) {
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
    IndexSampleGradInner<T, Context, int>(ctx, out_grad, index, x_grad);
  } else if (index_type == DataType::INT64) {
    IndexSampleGradInner<T, Context, int64_t>(ctx, out_grad, index, x_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_sample_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexSampleGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
