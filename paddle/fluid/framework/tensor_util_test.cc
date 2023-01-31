//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <cmath>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/isfinite_op.h"
namespace paddle {
namespace framework {

TEST(TensorCopy, Tensor) {
  phi::DenseTensor src_tensor;
  phi::DenseTensor dst_tensor;
  phi::CPUContext cpu_ctx((platform::CPUPlace()));

  int* src_ptr = src_tensor.mutable_data<int>(phi::make_ddim({3, 3}),
                                              platform::CPUPlace());

  int arr[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  memcpy(src_ptr, arr, 9 * sizeof(int));
  src_tensor.set_layout(DataLayout::kAnyLayout);

  auto cpu_place = new platform::CPUPlace();
  TensorCopy(src_tensor, *cpu_place, &dst_tensor);

  const int* dst_ptr = dst_tensor.data<int>();
  EXPECT_NE(src_ptr, dst_ptr);
  for (size_t i = 0; i < 9; ++i) {
    EXPECT_EQ(src_ptr[i], dst_ptr[i]);
  }

  TensorCopy(dst_tensor, *cpu_place, &dst_tensor);
  for (size_t i = 0; i < 9; ++i) {
    EXPECT_EQ(src_ptr[i], dst_ptr[i]);
  }

  EXPECT_TRUE(dst_tensor.layout() == src_tensor.layout());

  phi::DenseTensor slice_tensor = src_tensor.Slice(1, 2);
  TensorCopy(slice_tensor, *cpu_place, &dst_tensor);
  const int* slice_ptr = slice_tensor.data<int>();
  dst_ptr = dst_tensor.data<int>();
  EXPECT_NE(dst_ptr, slice_ptr);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(dst_ptr[i], slice_ptr[i]);
  }
  EXPECT_TRUE(dst_tensor.layout() == src_tensor.layout());

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    phi::DenseTensor src_tensor;
    phi::DenseTensor gpu_tensor;
    phi::DenseTensor dst_tensor;

    int* src_ptr = src_tensor.mutable_data<int>(phi::make_ddim({3, 3}),
                                                platform::CPUPlace());

    int arr[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    memcpy(src_ptr, arr, 9 * sizeof(int));

    // CPU phi::DenseTensor to GPU phi::DenseTensor
    auto gpu_place = new platform::CUDAPlace(0);
    phi::GPUContext gpu_ctx(*gpu_place);
    gpu_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                             .GetAllocator(*gpu_place, gpu_ctx.stream())
                             .get());
    gpu_ctx.PartialInitWithAllocator();
    TensorCopy(src_tensor, *gpu_place, gpu_ctx, &gpu_tensor);

    // GPU phi::DenseTensor to CPU phi::DenseTensor
    auto cpu_place = new platform::CPUPlace();
    TensorCopy(gpu_tensor, *cpu_place, gpu_ctx, &dst_tensor);

    // Sync before Compare Tensors
    gpu_ctx.Wait();
    const int* dst_ptr = dst_tensor.data<int>();
    EXPECT_NE(src_ptr, dst_ptr);
    for (size_t i = 0; i < 9; ++i) {
      EXPECT_EQ(src_ptr[i], dst_ptr[i]);
    }

    // Copy the same tensor
    TensorCopy(gpu_tensor, *gpu_place, gpu_ctx, &gpu_tensor);
    gpu_ctx.Wait();
    const int* dst_ptr_tmp = dst_tensor.data<int>();
    EXPECT_NE(src_ptr, dst_ptr_tmp);
    for (size_t i = 0; i < 9; ++i) {
      EXPECT_EQ(src_ptr[i], dst_ptr_tmp[i]);
    }

    phi::DenseTensor slice_tensor = src_tensor.Slice(1, 2);

    // CPU Slice phi::DenseTensor to GPU phi::DenseTensor
    TensorCopy(slice_tensor, *gpu_place, gpu_ctx, &gpu_tensor);

    // GPU phi::DenseTensor to CPU phi::DenseTensor
    TensorCopy(gpu_tensor, *cpu_place, gpu_ctx, &dst_tensor);

    // Sync before Compare Slice Tensors
    gpu_ctx.Wait();
    const int* slice_ptr = slice_tensor.data<int>();
    dst_ptr = dst_tensor.data<int>();
    EXPECT_NE(dst_ptr, slice_ptr);
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(dst_ptr[i], slice_ptr[i]);
    }

    EXPECT_TRUE(dst_tensor.layout() == src_tensor.layout());
  }
#endif
}

TEST(TensorFromVector, Tensor) {
  {
    std::vector<int> src_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    phi::DenseTensor cpu_tensor;

    // Copy to CPU phi::DenseTensor
    cpu_tensor.Resize(phi::make_ddim({3, 3}));
    auto cpu_place = new paddle::platform::CPUPlace();
    paddle::framework::TensorFromVector<int>(src_vec, &cpu_tensor);

    // Compare Tensors
    const int* cpu_ptr = cpu_tensor.data<int>();
    const int* src_ptr = src_vec.data();
    EXPECT_NE(src_ptr, cpu_ptr);
    for (size_t i = 0; i < 9; ++i) {
      EXPECT_EQ(src_ptr[i], cpu_ptr[i]);
    }

    src_vec.erase(src_vec.begin(), src_vec.begin() + 5);
    cpu_tensor.Resize(phi::make_ddim({2, 2}));
    paddle::framework::TensorFromVector<int>(src_vec, &cpu_tensor);
    cpu_ptr = cpu_tensor.data<int>();
    src_ptr = src_vec.data();
    EXPECT_NE(src_ptr, cpu_ptr);
    for (size_t i = 0; i < 5; ++i) {
      EXPECT_EQ(src_ptr[i], cpu_ptr[i]);
    }

    delete cpu_place;
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    std::vector<int> src_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    phi::DenseTensor cpu_tensor;
    phi::DenseTensor gpu_tensor;
    phi::DenseTensor dst_tensor;

    // Copy to CPU phi::DenseTensor
    cpu_tensor.Resize(phi::make_ddim({3, 3}));
    auto cpu_place = new paddle::platform::CPUPlace();
    phi::CPUContext cpu_ctx(*cpu_place);
    paddle::framework::TensorFromVector<int>(src_vec, cpu_ctx, &cpu_tensor);

    // Copy to GPUTensor
    gpu_tensor.Resize(phi::make_ddim({3, 3}));
    auto gpu_place = new paddle::platform::CUDAPlace();
    phi::GPUContext gpu_ctx(*gpu_place);
    gpu_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                             .GetAllocator(*gpu_place, gpu_ctx.stream())
                             .get());
    gpu_ctx.PartialInitWithAllocator();
    paddle::framework::TensorFromVector<int>(src_vec, gpu_ctx, &gpu_tensor);
    // Copy from GPU to CPU tensor for comparison
    paddle::framework::TensorCopy(gpu_tensor, *cpu_place, gpu_ctx, &dst_tensor);

    // Sync before Compare Tensors
    gpu_ctx.Wait();
    const int* src_ptr = src_vec.data();
    const int* cpu_ptr = cpu_tensor.data<int>();
    const int* dst_ptr = dst_tensor.data<int>();
    EXPECT_NE(src_ptr, cpu_ptr);
    EXPECT_NE(src_ptr, dst_ptr);
    for (size_t i = 0; i < 9; ++i) {
      EXPECT_EQ(src_ptr[i], cpu_ptr[i]);
      EXPECT_EQ(src_ptr[i], dst_ptr[i]);
    }

    src_vec.erase(src_vec.begin(), src_vec.begin() + 5);

    cpu_tensor.Resize(phi::make_ddim({2, 2}));
    paddle::framework::TensorFromVector<int>(src_vec, cpu_ctx, &cpu_tensor);
    gpu_tensor.Resize(phi::make_ddim({2, 2}));
    paddle::framework::TensorFromVector<int>(src_vec, gpu_ctx, &gpu_tensor);
    paddle::framework::TensorCopy(gpu_tensor, *cpu_place, gpu_ctx, &dst_tensor);

    // Sync before Compare Tensors
    gpu_ctx.Wait();
    src_ptr = src_vec.data();
    cpu_ptr = cpu_tensor.data<int>();
    dst_ptr = dst_tensor.data<int>();
    EXPECT_NE(src_ptr, cpu_ptr);
    EXPECT_NE(src_ptr, dst_ptr);
    for (size_t i = 0; i < 5; ++i) {
      EXPECT_EQ(src_ptr[i], cpu_ptr[i]);
      EXPECT_EQ(src_ptr[i], dst_ptr[i]);
    }

    delete cpu_place;
    delete gpu_place;
  }
#endif
}

TEST(TensorToVector, Tensor) {
  {
    phi::DenseTensor src;
    int* src_ptr = src.mutable_data<int>({3, 3}, paddle::platform::CPUPlace());
    for (int i = 0; i < 3 * 3; ++i) {
      src_ptr[i] = i;
    }

    paddle::platform::CPUPlace place;
    std::vector<int> dst;
    paddle::framework::TensorToVector<int>(src, &dst);

    for (int i = 0; i < 3 * 3; ++i) {
      EXPECT_EQ(src_ptr[i], dst[i]);
    }
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    std::vector<int> src_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    phi::DenseTensor gpu_tensor;
    paddle::platform::CUDAPlace place;
    phi::GPUContext gpu_ctx(place);
    gpu_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                             .GetAllocator(place, gpu_ctx.stream())
                             .get());
    gpu_ctx.PartialInitWithAllocator();
    paddle::framework::TensorFromVector<int>(src_vec, gpu_ctx, &gpu_tensor);

    std::vector<int> dst;
    paddle::framework::TensorToVector<int>(gpu_tensor, gpu_ctx, &dst);

    for (int i = 0; i < 3 * 3; ++i) {
      EXPECT_EQ(src_vec[i], dst[i]);
    }
  }
#endif
}

TEST(TensorToVector, Tensor_bool) {
  phi::DenseTensor src;
  bool* src_ptr = src.mutable_data<bool>({3, 3}, paddle::platform::CPUPlace());
  for (int i = 0; i < 3 * 3; ++i) {
    src_ptr[i] = static_cast<bool>(i % 2);
  }

  paddle::platform::CPUPlace place;
  std::vector<bool> dst;
  paddle::framework::TensorToVector<bool>(src, &dst);

  for (int i = 0; i < 3 * 3; ++i) {
    EXPECT_EQ(src_ptr[i], dst[i]);
  }

#ifdef PADDLE_WITH_CUDA
  {
    std::vector<bool> src_vec = {
        false,
        true,
        false,
        true,
        false,
        true,
        false,
        true,
        false,
    };
    phi::DenseTensor gpu_tensor;
    paddle::platform::CUDAPlace place;
    phi::GPUContext gpu_ctx(place);
    gpu_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                             .GetAllocator(place, gpu_ctx.stream())
                             .get());
    gpu_ctx.PartialInitWithAllocator();
    paddle::framework::TensorFromVector<bool>(src_vec, gpu_ctx, &gpu_tensor);

    std::vector<bool> dst;
    paddle::framework::TensorToVector<bool>(gpu_tensor, gpu_ctx, &dst);

    for (int i = 0; i < 3 * 3; ++i) {
      EXPECT_EQ(src_vec[i], dst[i]);
    }
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  {
    std::vector<bool> src_vec = {
        false,
        true,
        false,
        true,
        false,
        true,
        false,
        true,
        false,
    };
    phi::DenseTensor npu_tensor;
    paddle::platform::NPUPlace place(0);
    paddle::platform::NPUDeviceContext npu_ctx(place);
    paddle::framework::TensorFromVector<bool>(src_vec, npu_ctx, &npu_tensor);

    std::vector<bool> dst;
    paddle::framework::TensorToVector<bool>(npu_tensor, npu_ctx, &dst);

    for (int i = 0; i < 3 * 3; ++i) {
      EXPECT_EQ(src_vec[i], dst[i]);
    }
  }
#endif
}

TEST(TensorFromDLPack, Tensor) {
  {
    std::vector<int> src_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    phi::DenseTensor cpu_tensor;

    cpu_tensor.Resize(phi::make_ddim({3, 3}));
    paddle::platform::CPUPlace cpu_place;
    phi::CPUContext cpu_ctx(cpu_place);
    paddle::framework::TensorFromVector<int>(src_vec, cpu_ctx, &cpu_tensor);
    paddle::framework::DLPackTensor dlpack_tensor(cpu_tensor, 1);

    phi::DenseTensor dst_tensor;
    paddle::framework::TensorFromDLPack(dlpack_tensor, &dst_tensor);

    auto cpu_ptr = cpu_tensor.data<int>();
    auto src_ptr = dst_tensor.data<int>();
    EXPECT_NE(src_ptr, cpu_ptr);
    for (size_t i = 0; i < 9; ++i) {
      EXPECT_EQ(src_ptr[i], cpu_ptr[i]);
    }
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    std::vector<int> src_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    phi::DenseTensor cpu_tensor;
    phi::DenseTensor gpu_tensor;
    phi::DenseTensor dst_tensor;
    phi::DenseTensor gpu_tensor_from_dlpack;

    // Copy to CPU phi::DenseTensor
    cpu_tensor.Resize(phi::make_ddim({3, 3}));
    paddle::platform::CPUPlace cpu_place;
    phi::CPUContext cpu_ctx(cpu_place);
    paddle::framework::TensorFromVector<int>(src_vec, cpu_ctx, &cpu_tensor);

    // Copy to GPUTensor
    gpu_tensor.Resize(phi::make_ddim({3, 3}));
    paddle::platform::CUDAPlace gpu_place;
    auto& gpu_ctx =
        *paddle::platform::DeviceContextPool::Instance().GetByPlace(gpu_place);
    paddle::framework::TensorFromVector<int>(src_vec, gpu_ctx, &gpu_tensor);
    gpu_ctx.Wait();

    paddle::framework::DLPackTensor dlpack_tensor(gpu_tensor, 1);
    paddle::framework::TensorFromDLPack(dlpack_tensor, &gpu_tensor_from_dlpack);
    gpu_ctx.Wait();

    // Copy from GPU to CPU tensor for comparison
    paddle::framework::TensorCopy(
        gpu_tensor_from_dlpack, cpu_place, gpu_ctx, &dst_tensor);
    // Sync before Compare Tensors
    gpu_ctx.Wait();
    const int* src_ptr = src_vec.data();
    const int* cpu_ptr = cpu_tensor.data<int>();
    const int* dst_ptr = dst_tensor.data<int>();
    EXPECT_NE(src_ptr, cpu_ptr);
    EXPECT_NE(src_ptr, dst_ptr);
    for (size_t i = 0; i < 9; ++i) {
      EXPECT_EQ(src_ptr[i], cpu_ptr[i]);
      EXPECT_EQ(src_ptr[i], dst_ptr[i]);
    }
  }
#endif
}

TEST(TensorContainsNAN, CPU) {
  {
    phi::DenseTensor src;
    float* buf = src.mutable_data<float>({3}, paddle::platform::CPUPlace());
    buf[0] = 0.0;
    buf[1] = NAN;
    buf[2] = 0.0;
    EXPECT_TRUE(paddle::framework::TensorContainsNAN(src));
    buf[1] = 0.0;
    EXPECT_FALSE(paddle::framework::TensorContainsNAN(src));
  }

  {
    phi::DenseTensor src;
    paddle::platform::float16* buf =
        src.mutable_data<paddle::platform::float16>(
            {3}, paddle::platform::CPUPlace());
    buf[0] = 0.0;
    buf[1].x = 0x7fff;
    buf[2] = 0.0;
    EXPECT_TRUE(paddle::framework::TensorContainsNAN(src));
    buf[1] = 0.0;
    EXPECT_FALSE(paddle::framework::TensorContainsNAN(src));
  }
}

TEST(TensorContainsInf, CPU) {
  {
    phi::DenseTensor src;
    double* buf = src.mutable_data<double>({3}, paddle::platform::CPUPlace());
    buf[0] = 1.0;
    buf[1] = INFINITY;
    buf[2] = 0.0;
    EXPECT_TRUE(paddle::framework::TensorContainsInf(src));
    buf[1] = 1.0;
    EXPECT_FALSE(paddle::framework::TensorContainsInf(src));
  }

  {
    phi::DenseTensor src;
    paddle::platform::float16* buf =
        src.mutable_data<paddle::platform::float16>(
            {3}, paddle::platform::CPUPlace());
    buf[0] = 1.0;
    buf[1].x = 0x7c00;
    buf[2] = 0.0;
    EXPECT_TRUE(paddle::framework::TensorContainsInf(src));
    buf[1] = 1.0;
    EXPECT_FALSE(paddle::framework::TensorContainsInf(src));
  }
}

TEST(TensorIsfinite, CPU) {
  {
    phi::DenseTensor src, out;
    double* buf = src.mutable_data<double>({3}, paddle::platform::CPUPlace());
    buf[0] = 1.0;
    buf[1] = INFINITY;
    buf[2] = 0.0;
    paddle::framework::TensorIsfinite(src, &out);
    EXPECT_EQ(out.data<bool>()[0], false);
    buf[1] = 1.0;
    paddle::framework::TensorIsfinite(src, &out);
    EXPECT_EQ(out.data<bool>()[0], true);
  }

  {
    phi::DenseTensor src, out;
    double* buf = src.mutable_data<double>({3}, paddle::platform::CPUPlace());
    buf[0] = 1.0;
    buf[1] = NAN;
    buf[2] = 0.0;
    paddle::framework::TensorIsfinite(src, &out);
    EXPECT_EQ(out.data<bool>()[0], false);
    buf[1] = 1.0;
    paddle::framework::TensorIsfinite(src, &out);
    EXPECT_EQ(out.data<bool>()[0], true);
  }

  {
    phi::DenseTensor src, out;
    paddle::platform::float16* buf =
        src.mutable_data<paddle::platform::float16>(
            {3}, paddle::platform::CPUPlace());
    buf[0] = 1.0;
    buf[1].x = 0x7c00;
    buf[2] = 0.0;
    paddle::framework::TensorIsfinite(src, &out);
    EXPECT_EQ(out.data<bool>()[0], false);
    buf[1] = 1.0;
    paddle::framework::TensorIsfinite(src, &out);
    EXPECT_EQ(out.data<bool>()[0], true);
    buf[1].x = 0x7fff;
    paddle::framework::TensorIsfinite(src, &out);
    EXPECT_EQ(out.data<bool>()[0], false);
  }
}

TEST(Tensor, FromAndToStream) {
  phi::DenseTensor src_tensor;
  int array[6] = {1, 2, 3, 4, 5, 6};
  src_tensor.Resize({2, 3});
  int* src_ptr = src_tensor.mutable_data<int>(platform::CPUPlace());
  for (int i = 0; i < 6; ++i) {
    src_ptr[i] = array[i];
  }
  {
    phi::DenseTensor dst_tensor;
    auto place = new platform::CPUPlace();
    phi::CPUContext cpu_ctx(*place);
    std::ostringstream oss;
    TensorToStream(oss, src_tensor, cpu_ctx);

    std::istringstream iss(oss.str());
    TensorFromStream(iss, &dst_tensor, cpu_ctx);
    int* dst_ptr = dst_tensor.mutable_data<int>(platform::CPUPlace());
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(dst_ptr[i], array[i]);
    }
    EXPECT_EQ(dst_tensor.dims(), src_tensor.dims());
    delete place;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    phi::DenseTensor gpu_tensor;
    gpu_tensor.Resize({2, 3});
    phi::DenseTensor dst_tensor;

    auto gpu_place = new platform::CUDAPlace();
    phi::GPUContext gpu_ctx(*gpu_place);
    gpu_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                             .GetAllocator(*gpu_place, gpu_ctx.stream())
                             .get());
    gpu_ctx.PartialInitWithAllocator();

    TensorCopy(src_tensor, *gpu_place, gpu_ctx, &gpu_tensor);

    std::ostringstream oss;
    TensorToStream(oss, gpu_tensor, gpu_ctx);

    std::istringstream iss(oss.str());
    TensorFromStream(
        iss,
        &dst_tensor,
        *platform::DeviceContextPool::Instance().Get(platform::CPUPlace()));

    int* dst_ptr = dst_tensor.mutable_data<int>(platform::CPUPlace());
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(dst_ptr[i], array[i]);
    }
    delete gpu_place;
  }
#endif
}

}  // namespace framework
}  // namespace paddle
