// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_elementwise_get_kernel.h"
#include <cstdio>

#include "glog/logging.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "thrust/copy.h"
#include "thrust/detail/count.h"
#include "thrust/device_ptr.h"
#include "thrust/iterator/counting_iterator.h"

namespace phi {

namespace {
struct IsTrueFunctor {
  __device__ bool operator()(bool value) const { return value; }
};

template <typename T, typename OffsetCal>
struct IndexToValueFunctor {
  const T* input_data;
  OffsetCal offset_calc;

  // 构造函数需要按值捕获，以便传递给 device
  IndexToValueFunctor(const T* data, OffsetCal calc)
      : input_data(data), offset_calc(calc) {}

  // device 端的操作符重载
  // 输入：logical_index (例如 0, 1, 2, ... numel-1)
  // 输出：位于该逻辑索引处的输入张量的值
  __device__ T operator()(int64_t logical_index) const {
    // 使用 offset_calc 将逻辑索引转换为物理内存偏移
    int64_t physical_offset = offset_calc.get(logical_index)[0];
    // 从输入数据中获取实际值
    return input_data[physical_offset];
  }
};

}  // namespace
template <typename T, typename IndexT = int>
void GPUIndexElementwiseGetKernel(const phi::GPUContext& dev_ctx,
                                  const DenseTensor& input,
                                  const std::vector<const DenseTensor*> index,
                                  const std::vector<int64_t>& input_dims,
                                  const std::vector<int64_t>& input_strides,
                                  const std::vector<int64_t>& index_dims,
                                  const std::vector<int64_t>& index_stride,
                                  const int64_t slice_offset,
                                  DenseTensor* output) {
  if (index.size() == 1 && index[0]->dtype() == phi::DataType::BOOL) {
    PADDLE_ENFORCE_EQ(
        index[0]->strides(),
        ::common::make_ddim(index_stride),
        common::errors::InvalidArgument("Index tensor should be contiguous."));
    const bool* index_data = index[0]->data<bool>();
    const T* input_data =
        input.data<T>() + slice_offset / phi::SizeOf(input.dtype());
    thrust::device_ptr<const bool> d_index_ptr(
        const_cast<const bool*>(index_data));
    int64_t numel = index[0]->numel();
    auto true_count = thrust::count(d_index_ptr, d_index_ptr + numel, true);
    output->Resize(common::make_ddim({true_count}));
    dev_ctx.template Alloc<T>(output);

    if (true_count == 0) {
      return;
    }
    auto reverse_shape = std::vector(input_dims.rbegin(), input_dims.rend());

    auto offset_calc = funcs::make_offset_calculator<1>(
        static_cast<int32_t>(input_dims.size()),
        reverse_shape.data(),
        {
            {input_strides.rbegin(), input_strides.rend()},
        });

    IndexToValueFunctor<T, decltype(offset_calc)> functor(input_data,
                                                          offset_calc);
    auto counting_it = thrust::make_counting_iterator(static_cast<int64_t>(0));
    auto transform_it = thrust::make_transform_iterator(counting_it, functor);
    thrust::copy_if(transform_it,
                    transform_it + numel,
                    d_index_ptr,
                    thrust::device_ptr<T>(output->data<T>()),
                    IsTrueFunctor());

    // const T* input_data =
    //     input.data<T>() + slice_offset / phi::SizeOf(input.dtype());
    // T* output_data = output->data<T>();

    // DenseTensor counter_tensor;

    // constexpr int nt = 128;
    // constexpr int vt = 4;
    // const dim3 block(nt);
    // const dim3 grid((numel + block.x * vt - 1) / (block.x * vt));
    // auto stream = dev_ctx.stream();
    //         << ", " << grid.z << " and block: " << block.x << ", " << block.y
    //         << ", " << block.z;
    // funcs::index_elementwise_with_tensor_kernel<nt, vt>
    //     <<<grid, block, 0, stream>>>(numel, [=] __device__(int idx) {
    //       if (idx >= numel) return;
    //       if (index_data[idx]) {
    //         printf("index_data[%d] is true\n", idx);
    //         int64_t index = prefix_sum_data[idx];
    //         auto input_offset = offset_calc.get(idx)[0];
    //         output_data[index] = input_data[input_offset];
    //       }
    //     });

  } else {
    int64_t numel = 0;
    int64_t num_indices = 0;
    std::vector<int64_t> shape_tmp;
    std::vector<int64_t> stride_tmp;
    funcs::cal_shape_stride(index_dims, &num_indices, &shape_tmp, &stride_tmp);

    auto index_ptrs = funcs::GetIndexDataPtrs<IndexT>(index);

    auto sizes = std::array<int64_t, DDim::kMaxRank>{};
    auto strides = std::array<int64_t, DDim::kMaxRank>{};

    for (int64_t i = 0; i < num_indices; i++) {
      sizes[i] = index_dims[i];
      strides[i] = index_stride[i];
    }

    std::array<int64_t*, 3> strides_array;
    std::vector<int64_t> desired_shape;
    std::array<std::vector<int64_t>, 3> strides_vec;

    funcs::IndexGetStride<3>(input_dims,
                             input_strides,
                             phi::SizeOf(input.dtype()),
                             std::vector<int64_t>(),
                             std::vector<int64_t>(),
                             phi::SizeOf(input.dtype()),
                             shape_tmp,
                             stride_tmp,
                             phi::SizeOf(index[0]->dtype()),
                             &desired_shape,
                             &strides_array,
                             &numel,
                             strides_vec);
    auto offset_calc =
        funcs::make_offset_calculator_put<3>(desired_shape, strides_array);

    const int64_t N = output->numel();
    PADDLE_ENFORCE_GE(
        N, 0, common::errors::InvalidArgument("Output numel must >= 0"));
    PADDLE_ENFORCE_LE(
        N,
        std::numeric_limits<int32_t>::max(),
        common::errors::InvalidArgument("Output numel must <= INT32_MAX"));
    constexpr int nt = 128;
    constexpr int vt = 4;
    const dim3 block(nt);
    const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
    auto stream = dev_ctx.stream();

    using dtype = funcs::OpaqueType<sizeof(T)>;

    const char* in_ptr =
        reinterpret_cast<const char*>(input.data<T>()) + slice_offset;
    char* out_ptr = reinterpret_cast<char*>(output->data<T>());
    funcs::index_elementwise_with_tensor_kernel<nt, vt>
        <<<grid, block, 0, stream>>>(N, [=] __device__(int idx) {
          const auto offsets = offset_calc.get(idx);
          char* const out_data = out_ptr + offsets[0];
          const char* const in_data = in_ptr + offsets[1];

          int64_t offset = 0;
#pragma unroll
          for (int64_t i = 0; i < num_indices; i++) {
            int64_t index =
                *reinterpret_cast<int64_t*>(index_ptrs[i] + offsets[2]);
            if (index < 0) {
              index += sizes[i];
            }
            offset += index * strides[i];
          }

          *reinterpret_cast<dtype*>(out_data) =
              *reinterpret_cast<const dtype*>(in_data + offset);
        });
  }
}

template <typename T, typename Context>
void IndexElementwiseGetKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const std::vector<const DenseTensor*>& index,
                               const std::vector<int64_t>& input_dims,
                               const std::vector<int64_t>& input_strides,
                               const std::vector<int64_t>& index_dims,
                               const std::vector<int64_t>& index_stride,
                               const int64_t slice_offset,
                               const bool accumulate,
                               DenseTensor* out) {
  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(
      index_type == phi::DataType::INT64 ||
          (index_type == phi::DataType::BOOL && index.size() == 1),
      true,
      common::errors::InvalidArgument(
          "Index holds the wrong type, it holds [%s], but "
          "desires to be [%s] or bool.",
          index_type,
          phi::DataType::INT64));

  auto out_dims = out->dims();
  if (out_dims.size() > 0) {
    std::vector<int64_t> output_dims(input_dims);
    out->Resize(phi::make_ddim(output_dims));
  }

  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) return;

  GPUIndexElementwiseGetKernel<T, int64_t>(dev_ctx,
                                           x,
                                           index,
                                           input_dims,
                                           input_strides,
                                           index_dims,
                                           index_stride,
                                           slice_offset,
                                           out);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_elementwise_get,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexElementwiseGetKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
