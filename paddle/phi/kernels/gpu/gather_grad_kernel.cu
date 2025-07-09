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

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"
#include "paddle/phi/kernels/gather_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
namespace phi {
template <typename T, typename Context>
void PrintTensor(const std::string& name,
                 const phi::DenseTensor& tensor,
                 const Context& dev_ctx) {
  phi::DenseTensor tensor_cpu;
  if (tensor.place().GetType() == phi::AllocationType::GPU) {
    phi::Copy(dev_ctx, tensor, phi::CPUPlace(), true, &tensor_cpu);
  } else {
    tensor_cpu = tensor;
  }

  const T* data = tensor_cpu.data<T>();
  auto dims = tensor_cpu.dims();
  int64_t numel = tensor_cpu.numel();

  std::cout << name << " (shape: [";
  for (int i = 0; i < dims.size(); ++i) {
    std::cout << dims[i];
    if (i < dims.size() - 1) std::cout << ", ";
  }
  std::cout << "]):" << std::endl;

  std::vector<int64_t> strides(dims.size(), 1);
  for (int i = dims.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }

  std::function<void(int, int, std::vector<int>)> print_recursive;
  print_recursive = [&](int dim, int offset, std::vector<int> indices) {
    if (dim == dims.size()) {
      std::cout << data[offset];
      return;
    }
    std::cout << "[";
    for (int i = 0; i < dims[dim]; ++i) {
      indices.push_back(i);
      print_recursive(dim + 1, offset + i * strides[dim], indices);
      indices.pop_back();
      if (i < dims[dim] - 1) std::cout << ", ";
    }
    std::cout << "]";
  };

  print_recursive(0, 0, {});
  std::cout << std::endl << std::endl;
}

template <typename T, typename Context>
DenseTensor BroadcastIndex(const Context& dev_ctx,
                           const DenseTensor& index,
                           const DenseTensor& out_grad,
                           int axis) {
  DenseTensor processed_index;

  // Case 1: 0D scaler → Broadcast to out_grad.shape[axis]
  if (index.dims().size() == 0) {
    // Create target shape [1, ... , out_grad.shape[axis], ... , 1]
    std::vector<int64_t> target_shape(out_grad.dims().size(), 1);
    target_shape[axis] = out_grad.dims()[axis];

    DenseTensor reshaped;
    ReshapeKernel<Context>(dev_ctx, index, IntArray({1}), &reshaped);
    ExpandKernel<T, Context>(
        dev_ctx, reshaped, IntArray(target_shape), &processed_index);
    return processed_index;
  }

  // Case 2: 1D vectorize → Aligned non-axis dimensions
  if (index.dims().size() == 1) {
    // Create the shape after unsqueeze [... , 1, index_size, 1, ...]
    std::vector<int64_t> unsqueeze_shape(out_grad.dims().size(), 1);
    unsqueeze_shape[axis] = index.dims()[0];

    DenseTensor reshaped;
    ReshapeKernel<Context>(
        dev_ctx, index, IntArray(unsqueeze_shape), &reshaped);
    ExpandKernel<T, Context>(dev_ctx,
                             reshaped,
                             IntArray(common::vectorize(out_grad.dims())),
                             &processed_index);
    return processed_index;
  }

  return index;
}

template <typename T, typename Context>
void GatherGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& index,
                      const DenseTensor& out_grad,
                      const Scalar& axis,
                      DenseTensor* x_grad) {
  // x [4, 2], index [2, 0], out [2, 0], x_grad [4, 2]
  if (out_grad.numel() == 0) {
    if (x_grad) {
      phi::Full<T, Context>(
          dev_ctx, phi::IntArray(common::vectorize(x_grad->dims())), 0, x_grad);
    }
    return;
  }
  const auto& index_type = index.dtype();
  auto axis_v = axis.to<int>();
  if (axis_v < 0) {
    axis_v += static_cast<int>(x.dims().size());
  }

  if (axis_v != 0) {
    if (index_type == DataType::INT32) {
      phi::funcs::GatherV2GradCUDAFunction<T, int32_t>(
          &out_grad, &index, axis_v, x_grad, dev_ctx);
    } else if (index_type == DataType::INT64) {
      phi::funcs::GatherV2GradCUDAFunction<T, int64_t>(
          &out_grad, &index, axis_v, x_grad, dev_ctx);
    }
    return;
  }

  dev_ctx.template Alloc<T>(x_grad);
  phi::funcs::set_constant(dev_ctx, x_grad, static_cast<float>(0));
  if (out_grad.numel() == 0) {
    return;
  }

  if (index.numel() == x.dims()[axis_v]) {
    if (index_type == DataType::INT32) {
      DenseTensor index_int64 =
          phi::Cast<int32_t, Context>(dev_ctx, index, DataType::INT64);
      phi::funcs::GPUScatterAdd<T, int64_t>(
          dev_ctx, out_grad, index_int64, x_grad, axis_v);
    } else if (index_type == DataType::INT64) {
      phi::funcs::GPUScatterAdd<T, int64_t>(
          dev_ctx, out_grad, index, x_grad, axis_v);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The data type of Input(Index) of gather_grad must be int32 or int64 "
          "on GPU."));
    }
  } else {
    if (index_type == DataType::INT32) {
      phi::funcs::GPUScatterAssign<T, int>(
          dev_ctx, out_grad, index, x_grad, false);
    } else if (index_type == DataType::INT64) {
      phi::funcs::GPUScatterAssign<T, int64_t>(
          dev_ctx, out_grad, index, x_grad, false);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The data type of Input(Index) of gather_grad must be int32 or int64 "
          "on GPU."));
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(gather_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::GatherGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
