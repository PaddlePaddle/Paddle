/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"

namespace phi {
template <typename T, typename Context>
void StridedElementwiseCopyKernel(const Context& dev_ctx,
                                  const DenseTensor& input,
                                  const std::vector<int64_t>& out_dims,
                                  const std::vector<int64_t>& out_strides,
                                  int64_t out_offset,
                                  DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = common::make_ddim(out_strides);
  meta.dims = common::make_ddim(out_dims);
  meta.offset = out_offset;
  out->set_meta(meta);
  auto numel = out->numel();
  T* output_data = out->data<T>();
  PADDLE_ENFORCE_NOT_NULL(
      output_data,
      common::errors::InvalidArgument(
          "StridedElementwiseCopyKernel's out tensor must complete "
          "mutable data before call kernel."));

  const T* input_data = input.data<T>();

  if (numel == 1) {
#ifdef PADDLE_WITH_HIP
    hipMemcpy(output_data,
              input_data,
              phi::SizeOf(input.dtype()),
              hipMemcpyDeviceToDevice);
#else
    cudaMemcpy(output_data,
               input_data,
               phi::SizeOf(input.dtype()),
               cudaMemcpyDeviceToDevice);
#endif

    return;
  }

  bool can_expand = phi::funcs::CheckIsLastDimsMatch(input.dims(), out->dims());
  PADDLE_ENFORCE_EQ(can_expand || input.numel() == 1,
                    true,
                    common::errors::InvalidArgument(
                        "Input shape(%s) must expand to out shape(%s).",
                        input.dims(),
                        out->dims()));

  std::array<int64_t*, 2> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 2> strides_vec;

  funcs::CopyStride<2>(out_dims,
                       out_strides,
                       phi::SizeOf(out->dtype()),
                       common::vectorize<int64_t>(input.dims()),
                       common::vectorize<int64_t>(input.strides()),
                       phi::SizeOf(input.dtype()),
                       &desired_shape,
                       &strides_array,
                       &numel,
                       strides_vec);

  auto offset_calc =
      funcs::make_offset_calculator_put<2, true>(desired_shape, strides_array);

  constexpr int block_size = 128;
  constexpr int loop_size = 4;
  const dim3 block(block_size);
  const dim3 grid((numel + block.x * loop_size - 1) / (block.x * loop_size));
  auto stream = dev_ctx.stream();
  using dtype = funcs::OpaqueType<sizeof(T)>;
  const char* in_ptr = reinterpret_cast<const char*>(input.data<T>());
  char* out_ptr = reinterpret_cast<char*>(out->data<T>());
  funcs::index_elementwise_with_tensor_kernel<block_size, loop_size>
      <<<grid, block, 0, stream>>>(numel, [=] __device__(int idx) {
        const auto offsets = offset_calc.get(idx);
        char* const out_data = out_ptr + offsets[0];
        const char* const in_data = in_ptr + offsets[1];

        *reinterpret_cast<dtype*>(out_data) =
            *reinterpret_cast<const dtype*>(in_data);
      });
}

}  // namespace phi

PD_REGISTER_KERNEL(strided_elementwise_copy,
                   GPU,
                   ALL_LAYOUT,
                   phi::StridedElementwiseCopyKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
                   ::phi::dtype::float16,
                   ::phi::dtype::bfloat16,
                   ::phi::dtype::complex<float>,
                   ::phi::dtype::complex<double>,
                   ::phi::dtype::float8_e4m3fn,
                   ::phi::dtype::float8_e5m2) {}
