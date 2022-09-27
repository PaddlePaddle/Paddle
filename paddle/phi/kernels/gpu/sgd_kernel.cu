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

#include "paddle/phi/kernels/sgd_kernel.h"

#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename MT>
__global__ void SGDKernelMT(const T* param,
                            const T* grad,
                            const T* learning_rate,
                            const int num,
                            T* param_out,
                            const MT* master_param,
                            MT* master_param_out) {
  MT lr = static_cast<MT>(learning_rate[0]);
  CUDA_KERNEL_LOOP(i, num) {
    MT p_data = master_param ? master_param[i] : static_cast<MT>(param[i]);
    MT g_data = static_cast<MT>(grad[i]);
    p_data = p_data - lr * g_data;
    param_out[i] = static_cast<T>(p_data);
    if (master_param_out) {
      master_param_out[i] = p_data;
    }
  }
}

template <typename T>
__global__ void SparseSGDFunctorKernel(const T* selected_rows,
                                       const int64_t* rows,
                                       const T* learning_rate,
                                       T* tensor_out,
                                       int64_t row_numel,
                                       int64_t limit) {
  for (int64_t i = blockIdx.x; i < limit; i += gridDim.x) {
    const T* selected_rows_ptr = selected_rows + i * row_numel;
    T* tensor_out_ptr = tensor_out + rows[i] * row_numel;
    for (int64_t index = threadIdx.x; index < row_numel; index += blockDim.x) {
      // Since index in rows of SelectedRows can be duplicate, we have to use
      // Atomic Operation to avoid concurrent write error.
      paddle::platform::CudaAtomicAdd(
          tensor_out_ptr + index,
          -static_cast<T>(1.0) * learning_rate[0] * selected_rows_ptr[index]);
    }
  }
}

template <typename T, typename Context>
void SGDDenseKernel(const Context& dev_ctx,
                    const DenseTensor& param,
                    const DenseTensor& learning_rate,
                    const DenseTensor& grad,
                    const paddle::optional<DenseTensor>& master_param,
                    bool multi_precision,
                    DenseTensor* param_out,
                    DenseTensor* master_param_out) {
  using MPDType = typename paddle::operators::details::MPTypeTrait<T>::Type;
  // do check here
  // if (multi_precision) {
  //   bool has_master =
  //       ctx.HasInput("MasterParam") && ctx.HasOutput("MasterParamOut");

  // }
  const MPDType* master_in_data =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out_data =
      multi_precision
          ? master_param_out->mutable_data<MPDType>(dev_ctx.GetPlace())
          : nullptr;

  int block = 512;
  int grid = (param.numel() + block - 1) / block;

  SGDKernelMT<T, MPDType><<<grid, block, 0, dev_ctx.stream()>>>(
      param.data<T>(),
      grad.data<T>(),
      learning_rate.data<T>(),
      param.numel(),
      param_out->mutable_data<T>(dev_ctx.GetPlace()),
      master_in_data,
      master_out_data);
}

template <typename T, typename Context>
void SGDDenseParamSparseGradKernel(
    const Context& dev_ctx,
    const DenseTensor& param,
    const DenseTensor& learning_rate,
    const SelectedRows& grad,
    const paddle::optional<DenseTensor>& master_param,
    bool multi_precision,
    DenseTensor* param_out,
    DenseTensor* master_param_out) {
  using MPDType = typename paddle::operators::details::MPTypeTrait<T>::Type;
  // do some check here
  // if (multi_precision) {
  //   bool has_master =
  //       ctx.HasInput("MasterParam") && ctx.HasOutput("MasterParamOut");

  // }
  const MPDType* master_in_data =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out_data =
      multi_precision
          ? master_param_out->mutable_data<MPDType>(dev_ctx.GetPlace())
          : nullptr;

  PADDLE_ENFORCE_EQ(
      &param,
      param_out,
      phi::errors::InvalidArgument(
          "The input tensor Param of SgdOp should be equal with ParamOut "
          "if variable's type is SelectedRows."));

  auto in_height = grad.height();
  auto out_dims = param_out->dims();
  PADDLE_ENFORCE_EQ(in_height,
                    out_dims[0],
                    phi::errors::InvalidArgument(
                        "The input tensor Grad's height of SgdOp should be "
                        "equal with ParamOut's dims. But received Grad's "
                        "height [%s] and ParamOut's dims [%s]",
                        in_height,
                        out_dims[0]));

  auto& in_value = grad.value();
  auto& in_rows = grad.rows();

  int64_t in_row_numel = in_value.numel() / in_rows.size();
  PADDLE_ENFORCE_EQ(in_row_numel,
                    param_out->numel() / in_height,
                    phi::errors::InvalidArgument(
                        "The in_row_numel of SgdOp should be equal with "
                        "param_out's numel / in_height."));

  auto* in_data = in_value.data<T>();
  auto* out_data = param_out->data<T>();

  const int kThreadsPerBlock = 256;
  int thread_x = kThreadsPerBlock;
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);
  paddle::framework::MixVector<int64_t> mixv_in_rows(&in_rows);
  SparseSGDFunctorKernel<<<max_blocks, thread_x, 0, dev_ctx.stream()>>>(
      in_data,
      mixv_in_rows.CUDAData(dev_ctx.GetPlace()),
      learning_rate.data<T>(),
      out_data,
      in_row_numel,
      in_rows.size());
}

template <typename T, typename Context>
void SGDSparseParamSparseGradKernel(
    const Context& dev_ctx,
    const SelectedRows& param,
    const DenseTensor& learning_rate,
    const SelectedRows& grad,
    const paddle::optional<SelectedRows>& master_param,
    bool multi_precision,
    SelectedRows* param_out,
    SelectedRows* master_param_out) {
  PADDLE_THROW("not impl");
}

}  // namespace phi

PD_REGISTER_KERNEL(sgd,
                   GPU,
                   ALL_LAYOUT,
                   phi::SGDDenseKernel,
                   phi::dtype::float16,
                   float,
                   double) {}

PD_REGISTER_KERNEL(sgd_dense_param_sparse_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SGDDenseParamSparseGradKernel,
                   phi::dtype::float16,
                   float,
                   double) {}

PD_REGISTER_KERNEL(sgd_sparse_param_sparse_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SGDSparseParamSparseGradKernel,
                   phi::dtype::float16,
                   float,
                   double) {}
