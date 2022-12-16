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

#include <gtest/gtest.h>
#include <vector>
#include "glog/logging.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"

template <typename T>
struct AddTernary_1 {
  inline HOSTDEVICE T operator()(T a, T b, T c) const { return a + b + c; }
};

template <typename T>
struct AddTernary_2 {
  inline HOSTDEVICE T operator()(T a, T b, T c) const { return a + b + c; }
};

template <typename T>
struct AddTernary_3 {
  inline HOSTDEVICE T operator()(T a, T b, T c) const { return a + b + c; }
};

template <typename T>
void InitValue(T* data, size_t numel, const int val) {
  for (auto i = 0; i < numel; ++i) {
    data[i] = static_cast<T>(val);
  }
}

template <typename T, typename Func>
void TestCase(const phi::GPUContext& dev_ctx,
              const phi::DDim& dim1,
              const phi::DDim& dim2,
              const phi::DDim& dim3,
              const phi::DDim& dim_out,
              const size_t times,
              Func compute) {
  phi::DataType dtype = paddle::experimental::CppTypeToDataType<T>::Type();
  const auto alloc_cpu =
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace());
  const auto alloc_gpu =
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());

  auto in1 = std::make_shared<phi::DenseTensor>(
      alloc_cpu.get(),
      phi::DenseTensorMeta(dtype, dim1, phi::DataLayout::NCHW));
  auto in2 = std::make_shared<phi::DenseTensor>(
      alloc_cpu.get(),
      phi::DenseTensorMeta(dtype, dim2, phi::DataLayout::NCHW));
  auto in3 = std::make_shared<phi::DenseTensor>(
      alloc_cpu.get(),
      phi::DenseTensorMeta(dtype, dim3, phi::DataLayout::NCHW));
  InitValue(in1->data<T>(), in1->numel(), 1);
  InitValue(in2->data<T>(), in2->numel(), 1);
  InitValue(in3->data<T>(), in3->numel(), 1);

  auto d_in1 = std::make_shared<phi::DenseTensor>(
      alloc_gpu.get(),
      phi::DenseTensorMeta(dtype, dim1, phi::DataLayout::NCHW));
  auto d_in2 = std::make_shared<phi::DenseTensor>(
      alloc_gpu.get(),
      phi::DenseTensorMeta(dtype, dim2, phi::DataLayout::NCHW));
  auto d_in3 = std::make_shared<phi::DenseTensor>(
      alloc_gpu.get(),
      phi::DenseTensorMeta(dtype, dim3, phi::DataLayout::NCHW));
  auto d_out = std::make_shared<phi::DenseTensor>(
      alloc_gpu.get(),
      phi::DenseTensorMeta(dtype, dim_out, phi::DataLayout::NCHW));
  phi::Copy(dev_ctx, *in1.get(), phi::GPUPlace(), false, d_in1.get());
  phi::Copy(dev_ctx, *in2.get(), phi::GPUPlace(), false, d_in2.get());
  phi::Copy(dev_ctx, *in3.get(), phi::GPUPlace(), false, d_in3.get());

  std::vector<const phi::DenseTensor*> inputs{
      d_in1.get(), d_in2.get(), d_in3.get()};
  std::vector<phi::DenseTensor*> outputs{d_out.get()};
  for (int i = 0; i < times; ++i) {
    phi::funcs::BroadcastKernel<phi::ElementwiseType::kTernary, T, T>(
        dev_ctx, inputs, &outputs, -1, compute);
  }
  dev_ctx.Wait();
}

TEST(Broadcast, add) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto place = paddle::platform::CUDAPlace();
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* dev_ctx = static_cast<const phi::GPUContext*>(pool.GetByPlace(place));
  size_t times = 10;

  do {
    auto dim1 = phi::make_ddim({1, 2048, 3584});
    auto dim2 = phi::make_ddim({1, 2048, 1});
    auto dim3 = phi::make_ddim({1, 1, 3584});
    auto dim_out = phi::make_ddim({1, 2048, 3584});
    TestCase<float>(
        *dev_ctx, dim1, dim2, dim3, dim_out, times, AddTernary_1<float>());
    TestCase<phi::dtype::float16>(*dev_ctx,
                                  dim1,
                                  dim2,
                                  dim3,
                                  dim_out,
                                  times,
                                  AddTernary_1<phi::dtype::float16>());
    TestCase<phi::dtype::bfloat16>(*dev_ctx,
                                   dim1,
                                   dim2,
                                   dim3,
                                   dim_out,
                                   times,
                                   AddTernary_1<phi::dtype::bfloat16>());
  } while (0);

  do {
    auto dim1 = phi::make_ddim({1, 256, 4, 256, 256});
    auto dim2 = phi::make_ddim({1, 256, 1, 1, 256});
    auto dim3 = phi::make_ddim({1, 1, 4, 256, 256});
    auto dim_out = phi::make_ddim({1, 256, 4, 256, 256});
    TestCase<float>(
        *dev_ctx, dim1, dim2, dim3, dim_out, times, AddTernary_2<float>());
    TestCase<phi::dtype::float16>(*dev_ctx,
                                  dim1,
                                  dim2,
                                  dim3,
                                  dim_out,
                                  times,
                                  AddTernary_2<phi::dtype::float16>());
    TestCase<phi::dtype::bfloat16>(*dev_ctx,
                                   dim1,
                                   dim2,
                                   dim3,
                                   dim_out,
                                   times,
                                   AddTernary_2<phi::dtype::bfloat16>());
  } while (0);

  do {
    auto dim1 = phi::make_ddim({1, 256, 256});
    auto dim2 = phi::make_ddim({1, 1, 256});
    auto dim3 = phi::make_ddim({1, 256, 1});
    auto dim_out = phi::make_ddim({1, 256, 256});
    TestCase<float>(
        *dev_ctx, dim1, dim2, dim3, dim_out, times, AddTernary_3<float>());
    TestCase<phi::dtype::float16>(*dev_ctx,
                                  dim1,
                                  dim2,
                                  dim3,
                                  dim_out,
                                  times,
                                  AddTernary_3<phi::dtype::float16>());
    TestCase<phi::dtype::bfloat16>(*dev_ctx,
                                   dim1,
                                   dim2,
                                   dim3,
                                   dim_out,
                                   times,
                                   AddTernary_3<phi::dtype::bfloat16>());
  } while (0);
#endif
}
