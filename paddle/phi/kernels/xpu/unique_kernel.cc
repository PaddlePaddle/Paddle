// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <climits>
#include <numeric>
#include <utility>
#include <vector>

#include "paddle/phi/kernels/unique_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/core/visit_type.h"

namespace phi {

template <typename Context, typename T, typename IndexT>
void XPUFlattenUniqueKernelImpl(const Context& dev_ctx,
                                const DenseTensor& x,
                                bool return_index,
                                bool return_inverse,
                                bool return_counts,
                                DenseTensor* out,
                                DenseTensor* indices,
                                DenseTensor* index,
                                DenseTensor* counts) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto* x_data = x.data<T>();
  int64_t x_len = x.numel();
  int r = XPU_SUCCESS;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int64_t unique_len_cpu = 0;
  int64_t* unique_len_xpu = RAII_GUARD.alloc_l3_or_gm<int64_t>(1);
  if (x_len != 0) {
    r = xpu::unique_count<XPUType, IndexT>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x_data),
        unique_len_xpu,
        x_len,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        false);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "unique_count");
    memory_utils::Copy(phi::CPUPlace(),
                       &unique_len_cpu,
                       dev_ctx.GetPlace(),
                       unique_len_xpu,
                       sizeof(int64_t));
  }
  out->Resize(common::make_ddim({unique_len_cpu}));
  auto* out_data = dev_ctx.template Alloc<T>(out);
  IndexT* indices_data = nullptr;
  if (return_index) {
    indices->Resize(common::make_ddim({unique_len_cpu}));
    indices_data = dev_ctx.template Alloc<IndexT>(indices);
  }

  IndexT* inverse_data = nullptr;
  if (return_inverse) {
    index->Resize(common::make_ddim({x_len}));
    inverse_data = dev_ctx.template Alloc<IndexT>(index);
  }

  IndexT* counts_data = nullptr;
  if (return_counts) {
    counts->Resize(common::make_ddim({unique_len_cpu}));
    counts_data = dev_ctx.template Alloc<IndexT>(counts);
  }
  if (x_len == 0) {
    return;
  }
  r = xpu::unique_compute<XPUType, IndexT>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x_data),
      reinterpret_cast<XPUType*>(out_data),
      x_len,
      unique_len_cpu,
      indices_data,
      counts_data,
      inverse_data,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      false);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "unique_compute");
}

template <typename Context, typename T, typename IndexT>
void XPUDimUniqueKernelImpl(const Context& dev_ctx,
                            const DenseTensor& x,
                            bool return_index,
                            bool return_inverse,
                            bool return_counts,
                            int axis,
                            DenseTensor* out,
                            DenseTensor* indices,
                            DenseTensor* index,
                            DenseTensor* counts) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int r = xpu::SUCCESS;
  const auto* x_data = x.data<T>();
  auto* x_trans_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(x.numel());
  std::vector<int> permute(x.dims().size());
  std::iota(permute.begin(), permute.end(), 0);
  permute[axis] = 0;
  permute[0] = axis;
  if (axis != 0) {
    auto x_shape = common::vectorize<int>(x.dims());
    r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                reinterpret_cast<const XPUType*>(x_data),
                                x_trans_data,
                                x_shape,
                                permute);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  } else {
    r = xpu::copy<XPUType>(dev_ctx.x_context(),
                           reinterpret_cast<const XPUType*>(x_data),
                           x_trans_data,
                           x.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  }

  DDim x_trans_dims = x.dims();
  x_trans_dims[0] = x.dims()[axis];
  x_trans_dims[axis] = x.dims()[0];
  DDim x_trans_flat_dims = common::flatten_to_2d(x_trans_dims, 1);
  int64_t axis_len = x_trans_flat_dims[0];
  int64_t slice_size = x_trans_flat_dims[1];
  auto x_trans_flat_dims_vec = common::vectorize<int64_t>(x_trans_flat_dims);

  auto* sorted_axis_idx = RAII_GUARD.alloc_l3_or_gm<IndexT>(axis_len);
  auto* sort_in_tmp = RAII_GUARD.alloc_l3_or_gm<XPUType>(axis_len);
  auto* sort_out_tmp = RAII_GUARD.alloc_l3_or_gm<XPUType>(axis_len);
  auto* x_trans_tmp = RAII_GUARD.alloc_l3_or_gm<XPUType>(x.numel());
  auto* ori_idx_xpu = RAII_GUARD.alloc_l3_or_gm<IndexT>(axis_len);
  auto* ori_idx_xpu_tmp = RAII_GUARD.alloc_l3_or_gm<IndexT>(axis_len);
  auto* sort_offset = RAII_GUARD.alloc_l3_or_gm<IndexT>(axis_len);
  r = xpu::range<IndexT>(
      dev_ctx.x_context(), sort_offset, 0, slice_size, axis_len);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "range");

  r = xpu::range<IndexT>(dev_ctx.x_context(), ori_idx_xpu, 0, 1, axis_len);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "range");

  // radix sort
  for (int64_t i = slice_size - 1; i >= 0; --i) {
    r = xpu::paddle_gather<XPUType, IndexT>(dev_ctx.x_context(),
                                            x_trans_data + i,
                                            sort_offset,
                                            sort_in_tmp,
                                            {x.numel() - i},
                                            axis_len,
                                            0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");
    r = xpu::stable_sort<XPUType, IndexT>(dev_ctx.x_context(),
                                          sort_in_tmp,
                                          sort_out_tmp,
                                          sorted_axis_idx,
                                          1,
                                          axis_len,
                                          false);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "stable_sort");
    r = xpu::paddle_gather<XPUType, IndexT>(dev_ctx.x_context(),
                                            x_trans_data,
                                            sorted_axis_idx,
                                            x_trans_tmp,
                                            x_trans_flat_dims_vec,
                                            axis_len,
                                            0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");
    std::swap(x_trans_data, x_trans_tmp);

    r = xpu::paddle_gather<IndexT, IndexT>(dev_ctx.x_context(),
                                           ori_idx_xpu,
                                           sorted_axis_idx,
                                           ori_idx_xpu_tmp,
                                           {axis_len},
                                           axis_len,
                                           0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");
    std::swap(ori_idx_xpu, ori_idx_xpu_tmp);
  }

  // adjacent difference
  int64_t compare_num = (axis_len - 1) * slice_size;
  auto* compare_results = RAII_GUARD.alloc_l3_or_gm<bool>(compare_num);
  if (compare_num > 0) {
    r = xpu::broadcast_equal<XPUType>(dev_ctx.x_context(),
                                      x_trans_data + slice_size,
                                      x_trans_data,
                                      compare_results,
                                      {compare_num},
                                      {compare_num});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_equal");
  }

  std::vector<IndexT> unique_axis;
  std::vector<IndexT> indices_cpu;
  std::vector<IndexT> inverse_cpu(axis_len);
  std::vector<IndexT> counts_cpu;
  std::vector<IndexT> ori_idx_cpu(axis_len);
  memory_utils::Copy(phi::CPUPlace(),
                     ori_idx_cpu.data(),
                     dev_ctx.GetPlace(),
                     ori_idx_xpu,
                     sizeof(IndexT) * axis_len);
  unique_axis.push_back(0);
  indices_cpu.push_back(ori_idx_cpu[0]);
  inverse_cpu[ori_idx_cpu[0]] = 0;
  IndexT unique_len = 1;
  IndexT repeat_cnt = 1;
  if (axis_len > 1) {
    DenseTensor adj_identical_cpu;
    adj_identical_cpu.Resize({axis_len - 1});
    bool* adj_identical_cpu_data =
        dev_ctx.template HostAlloc<bool>(&adj_identical_cpu);
    auto* adj_identical_xpu = RAII_GUARD.alloc_l3_or_gm<bool>(axis_len - 1);
    r = xpu::reduce_all<bool>(dev_ctx.x_context(),
                              compare_results,
                              adj_identical_xpu,
                              {axis_len - 1, slice_size},
                              {1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_all");

    memory_utils::Copy(phi::CPUPlace(),
                       adj_identical_cpu_data,
                       dev_ctx.GetPlace(),
                       adj_identical_xpu,
                       (axis_len - 1) * sizeof(bool));

    for (IndexT i = 1; i < axis_len; ++i) {
      if (!adj_identical_cpu_data[i - 1]) {
        unique_axis.push_back(i);
        indices_cpu.push_back(ori_idx_cpu[i]);
        counts_cpu.push_back(repeat_cnt);
        ++unique_len;
        repeat_cnt = 1;
      } else {
        ++repeat_cnt;
      }
      inverse_cpu[ori_idx_cpu[i]] = unique_len - 1;
    }
  }
  counts_cpu.push_back(repeat_cnt);
  DDim out_dims = x.dims();
  out_dims[axis] = unique_len;
  out->Resize(out_dims);
  auto* out_data = dev_ctx.template Alloc<T>(out);

  auto* unique_axis_idx_xpu = RAII_GUARD.alloc_l3_or_gm<IndexT>(unique_len);
  auto* out_trans_data =
      RAII_GUARD.alloc_l3_or_gm<XPUType>(unique_len * slice_size);
  memory_utils::Copy(dev_ctx.GetPlace(),
                     unique_axis_idx_xpu,
                     phi::CPUPlace(),
                     unique_axis.data(),
                     unique_len * sizeof(IndexT));
  r = xpu::paddle_gather<XPUType, IndexT>(dev_ctx.x_context(),
                                          x_trans_data,
                                          unique_axis_idx_xpu,
                                          out_trans_data,
                                          x_trans_flat_dims_vec,
                                          unique_len,
                                          0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");
  DDim out_trans_dims = x_trans_dims;
  out_trans_dims[0] = unique_len;
  auto out_trans_dims_vec = common::vectorize<int>(out_trans_dims);
  if (axis != 0) {
    r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                out_trans_data,
                                reinterpret_cast<XPUType*>(out_data),
                                out_trans_dims_vec,
                                permute);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  } else {
    r = xpu::copy<XPUType>(dev_ctx.x_context(),
                           out_trans_data,
                           reinterpret_cast<XPUType*>(out_data),
                           unique_len * slice_size);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  }
  if (return_index) {
    indices->Resize({unique_len});
    auto* indices_data = dev_ctx.template Alloc<IndexT>(indices);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       indices_data,
                       phi::CPUPlace(),
                       indices_cpu.data(),
                       sizeof(IndexT) * unique_len);
  }

  if (return_inverse) {
    index->Resize({axis_len});
    auto* reverse_data = dev_ctx.template Alloc<IndexT>(index);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       reverse_data,
                       phi::CPUPlace(),
                       inverse_cpu.data(),
                       sizeof(IndexT) * axis_len);
  }

  if (return_counts) {
    counts->Resize({unique_len});
    auto* counts_data = dev_ctx.template Alloc<IndexT>(counts);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       counts_data,
                       phi::CPUPlace(),
                       counts_cpu.data(),
                       sizeof(IndexT) * unique_len);
  }
}

template <typename T, typename Context>
void UniqueKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  bool return_index,
                  bool return_inverse,
                  bool return_counts,
                  const std::vector<int>& axis,
                  DataType dtype,
                  DenseTensor* out,
                  DenseTensor* indices,
                  DenseTensor* index,
                  DenseTensor* counts) {
  bool is_sorted = true;
  UniqueRawKernel<T, Context>(dev_ctx,
                              x,
                              return_index,
                              return_inverse,
                              return_counts,
                              axis,
                              dtype,
                              is_sorted,
                              out,
                              indices,
                              index,
                              counts);
}

template <typename T, typename Context>
void UniqueRawKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     bool return_index,
                     bool return_inverse,
                     bool return_counts,
                     const std::vector<int>& axis,
                     DataType dtype,
                     bool is_sorted,
                     DenseTensor* out,
                     DenseTensor* indices,
                     DenseTensor* index,
                     DenseTensor* counts) {
  if (dtype == DataType::INT32) {
    PADDLE_ENFORCE_LE(
        x.numel(),
        INT_MAX,
        common::errors::InvalidArgument(
            "The number of elements in Input(X) should be less than or "
            "equal to INT_MAX, but received num is %d. Please set `dtype` to "
            "int64.",
            x.numel()));
  }

  if (axis.empty()) {
    PD_VISIT_BASE_INTEGRAL_TYPES(dtype, "XPUFlattenUniqueKernelImpl", [&] {
      XPUFlattenUniqueKernelImpl<Context, T, data_t>(dev_ctx,
                                                     x,
                                                     return_index,
                                                     return_inverse,
                                                     return_counts,
                                                     out,
                                                     indices,
                                                     index,
                                                     counts);
    });
  } else {
    int axis_value = axis[0];
    axis_value = (axis_value == -1) ? (x.dims().size() - 1) : axis_value;
    PD_VISIT_BASE_INTEGRAL_TYPES(dtype, "XPUDimUniqueKernelImpl", [&] {
      XPUDimUniqueKernelImpl<Context, T, data_t>(dev_ctx,
                                                 x,
                                                 return_index,
                                                 return_inverse,
                                                 return_counts,
                                                 axis_value,
                                                 out,
                                                 indices,
                                                 index,
                                                 counts);
    });
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    unique, XPU, ALL_LAYOUT, phi::UniqueKernel, float, int, int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(
    unique_raw, XPU, ALL_LAYOUT, phi::UniqueRawKernel, float, int, int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
}
