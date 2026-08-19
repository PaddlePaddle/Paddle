// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/put_along_axis_kernel.h"

#include <algorithm>
#include <type_traits>
#include <vector>

#include "paddle/common/layout.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/put_along_axis_grad_kernel.h"

namespace phi {

int64_t get_reduction_mode(const std::string& reduce) {
  if (reduce == "assign") {
    return 0;
  } else if (reduce == "add") {
    return 1;
  } else if (reduce == "multiply" || reduce == "mul") {
    return 2;
  } else if (reduce == "mean") {
    return 3;
  } else if (reduce == "amax") {
    return 4;
  } else if (reduce == "amin") {
    return 5;
  } else {
    PADDLE_THROW(errors::InvalidArgument(
        "can not support reduce: '%s' for put_along_axis kernel, only "
        "support reduce op: 'add', 'assign', 'mul', 'mean', 'amin', 'amax' and "
        "'multiply', the "
        "default reduce "
        "op is 'assign' ",
        reduce));
  }
}

template <typename T, typename Context>
void PutAlongAxisKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& index,
                        const DenseTensor& value,
                        int axis,
                        const std::string& reduce,
                        bool include_self,
                        DenseTensor* out) {
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0 || index.numel() == 0) return;

  const auto& index_dtype = index.dtype();
  bool index_dtype_match =
      index_dtype == DataType::INT32 || index_dtype == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_dtype_match,
                    true,
                    errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        DataTypeToString(index_dtype),
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64)));

  auto input_dtype = x.dtype();
  std::vector<int64_t> x_shape = vectorize<int64_t>(x.dims());
  std::vector<int64_t> index_shape = vectorize<int64_t>(index.dims());
  std::vector<int64_t> value_shape = vectorize<int64_t>(value.dims());
  using XPUType = typename XPUTypeTrait<T>::Type;
  int64_t reduce_mode = get_reduction_mode(reduce);

  bool invalid_input =
      (input_dtype == DataType::INT32 || input_dtype == DataType::INT64) &&
      (!include_self || reduce_mode > 2);
  PADDLE_ENFORCE_EQ(invalid_input,
                    false,
                    errors::InvalidArgument(
                        "Only support include_self = true and reduce mode: "
                        "'add', 'assign' and 'multiply' for int32/int64"));

  PADDLE_ENFORCE_EQ(index.dims().size(),
                    value.dims().size(),
                    errors::InvalidArgument(
                        "The input(Index) and the input(Value) must have same "
                        "rank, but received Index rank is %d, Value rank is %d",
                        index.dims().size(),
                        value.dims().size()));

  if (index_dtype == DataType::INT32) {
    int ret = xpu::paddle_put_along_axis(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        reinterpret_cast<const XPUType*>(value.data<T>()),
        index.data<int32_t>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        x_shape,
        value_shape,
        index_shape,
        axis,
        reduce_mode,
        include_self);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "paddle_put_along_axis");

  } else {
    int ret = xpu::paddle_put_along_axis(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        reinterpret_cast<const XPUType*>(value.data<T>()),
        index.data<int64_t>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        x_shape,
        value_shape,
        index_shape,
        axis,
        reduce_mode,
        include_self);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "paddle_put_along_axis");
  }
}

namespace {

inline std::vector<int64_t> GetStrides(const DDim& dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

template <typename IndexT>
int64_t GetPutOffset(const IndexT* index_data,
                     int64_t value_offset,
                     const DDim& index_dims,
                     const DDim& x_dims,
                     const std::vector<int64_t>& x_strides,
                     int axis) {
  int64_t remaining = value_offset;
  int64_t x_offset = 0;
  for (int i = static_cast<int>(index_dims.size()) - 1; i >= 0; --i) {
    const int64_t coord = remaining % index_dims[i];
    remaining /= index_dims[i];
    if (i == axis) {
      int64_t put_index = static_cast<int64_t>(index_data[value_offset]);
      if (put_index < 0) {
        put_index += x_dims[axis];
      }
      x_offset += put_index * x_strides[i];
    } else {
      x_offset += coord * x_strides[i];
    }
  }
  return x_offset;
}

template <typename T, typename SrcT>
void CastHostTensorData(const DenseTensor& src, DenseTensor* dst) {
  const auto* src_data = src.data<SrcT>();
  auto* dst_data = dst->data<T>();
  for (int64_t i = 0; i < src.numel(); ++i) {
    dst_data[i] = static_cast<T>(src_data[i]);
  }
}

template <typename T>
void CastHostTensorToKernelType(const DenseTensor& src, DenseTensor* dst) {
  if (src.dtype() == DataType::INT64) {
    CastHostTensorData<T, int64_t>(src, dst);
  } else {
    PADDLE_THROW(errors::InvalidArgument(
        "Unsupported out_grad dtype %s for XPU put_along_axis_grad host path",
        DataTypeToString(src.dtype())));
  }
}

template <typename T, typename IndexT>
void ComputePutAlongAxisGradOnHost(const DenseTensor& x,
                                   const DenseTensor& index,
                                   const DenseTensor& value,
                                   const DenseTensor& out,
                                   const DenseTensor& out_grad,
                                   int axis,
                                   const std::string& reduce,
                                   bool include_self,
                                   DenseTensor* x_grad,
                                   DenseTensor* value_grad) {
  const auto x_dims = x.dims();
  const auto index_dims = index.dims();
  if (axis < 0) {
    axis += x_dims.size();
  }

  const auto x_strides = GetStrides(x_dims);
  const int64_t index_numel = index.numel();
  const auto* index_data = index.data<IndexT>();
  const auto* x_data = x.data<T>();
  const auto* value_data = value.data<T>();
  const auto* out_data = out.data<T>();
  const auto* out_grad_data = out_grad.data<T>();

  if (x_grad) {
    auto* x_grad_data = x_grad->data<T>();
    std::copy(out_grad_data, out_grad_data + out_grad.numel(), x_grad_data);

    if (!include_self || reduce == "assign") {
      for (int64_t i = 0; i < index_numel; ++i) {
        x_grad_data[GetPutOffset(
            index_data, i, index_dims, x_dims, x_strides, axis)] =
            static_cast<T>(0);
      }
    } else if (reduce == "multiply" || reduce == "mul") {
      std::vector<int64_t> last_visit(x.numel(), -1);
      for (int64_t i = 0; i < index_numel; ++i) {
        last_visit[GetPutOffset(
            index_data, i, index_dims, x_dims, x_strides, axis)] = i;
      }
      for (int64_t i = 0; i < index_numel; ++i) {
        const int64_t x_offset =
            GetPutOffset(index_data, i, index_dims, x_dims, x_strides, axis);
        if (last_visit[x_offset] == i) {
          if (x_data[x_offset] != static_cast<T>(0)) {
            x_grad_data[x_offset] =
                x_grad_data[x_offset] * out_data[x_offset] / x_data[x_offset];
          } else {
            x_grad_data[x_offset] = static_cast<T>(0);
          }
        }
      }
    } else if (reduce == "amin" || reduce == "amax") {
      std::vector<int> num_elements(x.numel(), 0);
      for (int64_t i = 0; i < index_numel; ++i) {
        int64_t x_offset =
            GetPutOffset(index_data, i, index_dims, x_dims, x_strides, axis);
        if (out_data[x_offset] != x_data[x_offset]) {
          x_grad_data[x_offset] = static_cast<T>(0);
        } else if (out_data[x_offset] == value_data[i]) {
          num_elements[x_offset] += 1;
        }
      }
      for (int64_t i = 0; i < x.numel(); ++i) {
        x_grad_data[i] = x_grad_data[i] / static_cast<T>(num_elements[i] + 1);
      }
    } else if (reduce == "mean") {
      std::vector<int> num_elements(x.numel(), 0);
      for (int64_t i = 0; i < index_numel; ++i) {
        num_elements[GetPutOffset(
            index_data, i, index_dims, x_dims, x_strides, axis)] += 1;
      }
      for (int64_t i = 0; i < x.numel(); ++i) {
        if (num_elements[i]) {
          x_grad_data[i] = x_grad_data[i] / static_cast<T>(num_elements[i] + 1);
        }
      }
    }
  }

  if (value_grad) {
    auto* value_grad_data = value_grad->data<T>();
    std::fill(value_grad_data,
              value_grad_data + value_grad->numel(),
              static_cast<T>(0));

    if (reduce == "assign") {
      std::vector<bool> used(x.numel(), false);
      for (int64_t i = index_numel - 1; i >= 0; --i) {
        int64_t x_offset =
            GetPutOffset(index_data, i, index_dims, x_dims, x_strides, axis);
        if (!used[x_offset]) {
          value_grad_data[i] = out_grad_data[x_offset];
          used[x_offset] = true;
        }
      }
    } else if (reduce == "add") {
      for (int64_t i = index_numel - 1; i >= 0; --i) {
        value_grad_data[i] = out_grad_data[GetPutOffset(
            index_data, i, index_dims, x_dims, x_strides, axis)];
      }
    } else if (reduce == "mean") {
      std::vector<int> num_elements(x.numel(), static_cast<int>(include_self));
      for (int64_t i = index_numel - 1; i >= 0; --i) {
        num_elements[GetPutOffset(
            index_data, i, index_dims, x_dims, x_strides, axis)] += 1;
      }
      for (int64_t i = index_numel - 1; i >= 0; --i) {
        int64_t x_offset =
            GetPutOffset(index_data, i, index_dims, x_dims, x_strides, axis);
        value_grad_data[i] =
            out_grad_data[x_offset] / static_cast<T>(num_elements[x_offset]);
      }
    } else if (reduce == "multiply" || reduce == "mul") {
      for (int64_t i = 0; i < index_numel; ++i) {
        int64_t x_offset =
            GetPutOffset(index_data, i, index_dims, x_dims, x_strides, axis);
        if (value_data[i] != static_cast<T>(0)) {
          value_grad_data[i] =
              out_grad_data[x_offset] * (out_data[x_offset] / value_data[i]);
        }
      }
    } else if (reduce == "amin" || reduce == "amax") {
      std::vector<int> num_elements(x.numel(), 0);
      for (int64_t i = 0; i < index_numel; ++i) {
        int64_t x_offset =
            GetPutOffset(index_data, i, index_dims, x_dims, x_strides, axis);
        if (out_data[x_offset] == value_data[i]) {
          num_elements[x_offset] += 1;
        }
      }
      for (int64_t i = 0; i < index_numel; ++i) {
        int64_t x_offset =
            GetPutOffset(index_data, i, index_dims, x_dims, x_strides, axis);
        if (out_data[x_offset] == value_data[i]) {
          int divisor = num_elements[x_offset];
          if (out_data[x_offset] == x_data[x_offset]) {
            divisor += 1;
          }
          value_grad_data[i] =
              out_grad_data[x_offset] / static_cast<T>(divisor);
        }
      }
    }
  }
}

}  // namespace

template <typename T, typename Context>
void PutAlongAxisGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& index,
                            const DenseTensor& value,
                            const DenseTensor& out,
                            const DenseTensor& out_grad,
                            int axis,
                            const std::string& reduce,
                            bool include_self,
                            DenseTensor* x_grad,
                            DenseTensor* value_grad) {
  if (x.numel() == 0) {
    if (x_grad) {
      dev_ctx.template Alloc<T>(x_grad);
    }
    if (value_grad) {
      dev_ctx.template Alloc<T>(value_grad);
    }
    return;
  }

  const auto& index_dtype = index.dtype();
  bool index_dtype_match =
      index_dtype == DataType::INT32 || index_dtype == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_dtype_match,
                    true,
                    errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        DataTypeToString(index_dtype),
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64)));

  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, phi::float16> ||
                std::is_same_v<T, phi::bfloat16>) {
    if ((reduce == "assign" || reduce.empty()) && index.numel() > 1000000) {
      using XPUType = typename XPUTypeTrait<T>::Type;
      std::vector<int64_t> x_shape = vectorize<int64_t>(x.dims());
      std::vector<int64_t> index_shape = vectorize<int64_t>(index.dims());
      if (x_grad) {
        Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
        DenseTensor zero_value;
        zero_value.Resize(index.dims());
        dev_ctx.template Alloc<T>(&zero_value);
        int ret =
            xpu::constant(dev_ctx.x_context(),
                          reinterpret_cast<XPUType*>(zero_value.data<T>()),
                          zero_value.numel(),
                          XPUType(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(ret, "constant");
        if (index_dtype == DataType::INT32) {
          ret = xpu::paddle_put_along_axis(
              dev_ctx.x_context(),
              reinterpret_cast<const XPUType*>(x_grad->data<T>()),
              reinterpret_cast<const XPUType*>(zero_value.data<T>()),
              index.data<int32_t>(),
              reinterpret_cast<XPUType*>(x_grad->data<T>()),
              x_shape,
              index_shape,
              index_shape,
              axis,
              0,
              true);
        } else {
          ret = xpu::paddle_put_along_axis(
              dev_ctx.x_context(),
              reinterpret_cast<const XPUType*>(x_grad->data<T>()),
              reinterpret_cast<const XPUType*>(zero_value.data<T>()),
              index.data<int64_t>(),
              reinterpret_cast<XPUType*>(x_grad->data<T>()),
              x_shape,
              index_shape,
              index_shape,
              axis,
              0,
              true);
        }
        PADDLE_ENFORCE_XDNN_SUCCESS(ret, "paddle_put_along_axis");
      }
      if (value_grad) {
        value_grad->Resize(index.dims());
        dev_ctx.template Alloc<T>(value_grad);
        int ret = 0;
        if (index_dtype == DataType::INT32) {
          ret = xpu::gather<XPUType, int32_t>(
              dev_ctx.x_context(),
              reinterpret_cast<const XPUType*>(out_grad.data<T>()),
              index.data<int32_t>(),
              reinterpret_cast<XPUType*>(value_grad->data<T>()),
              x_shape,
              index_shape,
              axis);
        } else {
          ret = xpu::gather<XPUType, int64_t>(
              dev_ctx.x_context(),
              reinterpret_cast<const XPUType*>(out_grad.data<T>()),
              index.data<int64_t>(),
              reinterpret_cast<XPUType*>(value_grad->data<T>()),
              x_shape,
              index_shape,
              axis);
        }
        PADDLE_ENFORCE_XDNN_SUCCESS(ret, "gather");
      }
      return;
    }
  }

  DenseTensor x_cpu;
  DenseTensor index_cpu;
  DenseTensor value_cpu;
  DenseTensor out_cpu;
  DenseTensor out_grad_cpu;
  Copy(dev_ctx, x, CPUPlace(), false, &x_cpu);
  Copy(dev_ctx, index, CPUPlace(), false, &index_cpu);
  Copy(dev_ctx, value, CPUPlace(), false, &value_cpu);
  Copy(dev_ctx, out, CPUPlace(), false, &out_cpu);
  Copy(dev_ctx, out_grad, CPUPlace(), false, &out_grad_cpu);

  const DenseTensor* out_grad_host = &out_grad_cpu;
  DenseTensor out_grad_cast_cpu;
  if (out_grad_cpu.dtype() != phi::CppTypeToDataType<T>::Type()) {
    out_grad_cast_cpu.Resize(out_grad_cpu.dims());
    dev_ctx.template HostAlloc<T>(&out_grad_cast_cpu);
    CastHostTensorToKernelType<T>(out_grad_cpu, &out_grad_cast_cpu);
    out_grad_host = &out_grad_cast_cpu;
  }

  DenseTensor x_grad_cpu;
  DenseTensor value_grad_cpu;
  if (x_grad) {
    x_grad_cpu.Resize(x.dims());
    dev_ctx.template HostAlloc<T>(&x_grad_cpu);
  }
  if (value_grad) {
    value_grad->Resize(index.dims());
    value_grad_cpu.Resize(index.dims());
    dev_ctx.template HostAlloc<T>(&value_grad_cpu);
  }

  if (index_dtype == DataType::INT32) {
    ComputePutAlongAxisGradOnHost<T, int32_t>(
        x_cpu,
        index_cpu,
        value_cpu,
        out_cpu,
        *out_grad_host,
        axis,
        reduce,
        include_self,
        x_grad ? &x_grad_cpu : nullptr,
        value_grad ? &value_grad_cpu : nullptr);
  } else {
    ComputePutAlongAxisGradOnHost<T, int64_t>(
        x_cpu,
        index_cpu,
        value_cpu,
        out_cpu,
        *out_grad_host,
        axis,
        reduce,
        include_self,
        x_grad ? &x_grad_cpu : nullptr,
        value_grad ? &value_grad_cpu : nullptr);
  }

  if (x_grad) {
    Copy(dev_ctx, x_grad_cpu, dev_ctx.GetPlace(), false, x_grad);
  }
  if (value_grad) {
    Copy(dev_ctx, value_grad_cpu, dev_ctx.GetPlace(), false, value_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(put_along_axis,
                   XPU,
                   ALL_LAYOUT,
                   phi::PutAlongAxisKernel,
                   float,
                   int64_t,
                   int,
                   phi::float16,
                   phi::bfloat16) {}

PD_REGISTER_KERNEL(put_along_axis_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::PutAlongAxisGradKernel,
                   float,
                   int64_t,
                   int,
                   int16_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {}
