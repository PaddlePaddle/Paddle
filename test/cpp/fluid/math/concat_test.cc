/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"

/**
 * case 1:
 *    inputs:
 *        t_a.shape: [2, 3, 4]
 *        t_b.shape: [3, 3, 4]
 *    output:
 *        out.shape: [5, 3, 4]
 */
template <typename DeviceContext, typename Place>
void ConcatCase1(DeviceContext* context) {
  phi::DenseTensor input_a_cpu;
  phi::DenseTensor input_b_cpu;
  phi::DenseTensor out_cpu;

  phi::DenseTensor input_a;
  phi::DenseTensor input_b;
  phi::DenseTensor out;

  auto dim_a = common::make_ddim({2, 3, 4});
  auto dim_b = common::make_ddim({3, 3, 4});
  auto dim_out = common::make_ddim({5, 3, 4});

  input_a.mutable_data<int>(dim_a, Place());
  input_b.mutable_data<int>(dim_b, Place());
  out.mutable_data<int>(dim_out, Place());

  if (phi::is_gpu_place(Place())) {
    input_a_cpu.mutable_data<int>(dim_a, phi::CPUPlace());
    input_b_cpu.mutable_data<int>(dim_b, phi::CPUPlace());
    out_cpu.mutable_data<int>(dim_out, phi::CPUPlace());
  }

  int* a_ptr = nullptr;
  int* b_ptr = nullptr;
  if (phi::is_gpu_place(Place())) {
    a_ptr = input_a_cpu.data<int>();
    b_ptr = input_b_cpu.data<int>();
  } else {
    a_ptr = input_a.data<int>();
    b_ptr = input_b.data<int>();
  }

  for (int i = 0; i < 2 * 3 * 4; ++i) {
    a_ptr[i] = i;
  }
  for (int i = 0; i < 3 * 3 * 4; ++i) {
    b_ptr[i] = i;
  }

  if (phi::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(input_a_cpu, Place(), &input_a);
    paddle::framework::TensorCopySync(input_b_cpu, Place(), &input_b);
  }

  std::vector<phi::DenseTensor> input;
  input.push_back(input_a);
  input.push_back(input_b);

  phi::funcs::ConcatFunctor<DeviceContext, int> concat_functor;
  concat_functor(*context, input, 0, &out);

  // check the dim of input_a, input_b
  PADDLE_ENFORCE_EQ(input_a.dims(),
                    dim_a,
                    common::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_a.dims(),
                        dim_a));
  PADDLE_ENFORCE_EQ(input_b.dims(),
                    dim_b,
                    common::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_b.dims(),
                        dim_b));

  int* out_ptr = nullptr;
  if (phi::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(out, phi::CPUPlace(), &out_cpu);
    out_ptr = out_cpu.data<int>();
  } else {
    out_ptr = out.data<int>();
  }

  int cols = 2 * 3 * 4;
  int idx_a = 0, idx_b = 0;
  for (int j = 0; j < 5 * 3 * 4; ++j) {
    if (j >= cols) {
      PADDLE_ENFORCE_EQ(out_ptr[j],
                        b_ptr[idx_b],
                        common::errors::InvalidArgument(
                            "Concat test failed, the result should be equal."));
      ++idx_b;
    } else {
      PADDLE_ENFORCE_EQ(out_ptr[j],
                        a_ptr[idx_a],
                        common::errors::InvalidArgument(
                            "Concat test failed, the result should be equal."));
      ++idx_a;
    }
  }
}

/**
 * case 2:
 *    inputs:
 *        t_a.shape: [2, 3, 4]
 *        t_b.shape: [2, 4, 4]
 *    output:
 *        out.shape: [2, 7, 4]
 */
template <typename DeviceContext, typename Place>
void ConcatCase2(DeviceContext* context) {
  phi::DenseTensor input_a_cpu;
  phi::DenseTensor input_b_cpu;
  phi::DenseTensor out_cpu;

  phi::DenseTensor input_a;
  phi::DenseTensor input_b;
  phi::DenseTensor out;

  auto dim_a = common::make_ddim({2, 3, 4});
  auto dim_b = common::make_ddim({2, 4, 4});
  auto dim_out = common::make_ddim({2, 7, 4});

  input_a.mutable_data<int>(dim_a, Place());
  input_b.mutable_data<int>(dim_b, Place());
  out.mutable_data<int>(dim_out, Place());

  if (phi::is_gpu_place(Place())) {
    input_a_cpu.mutable_data<int>(dim_a, phi::CPUPlace());
    input_b_cpu.mutable_data<int>(dim_b, phi::CPUPlace());
    out_cpu.mutable_data<int>(dim_out, phi::CPUPlace());
  }

  int* a_ptr = nullptr;
  int* b_ptr = nullptr;
  if (phi::is_gpu_place(Place())) {
    a_ptr = input_a_cpu.data<int>();
    b_ptr = input_b_cpu.data<int>();
  } else {
    a_ptr = input_a.data<int>();
    b_ptr = input_b.data<int>();
  }

  for (int i = 0; i < 2 * 3 * 4; ++i) {
    a_ptr[i] = i;
  }
  for (int i = 0; i < 2 * 4 * 4; ++i) {
    b_ptr[i] = i;
  }

  if (phi::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(input_a_cpu, Place(), &input_a);
    paddle::framework::TensorCopySync(input_b_cpu, Place(), &input_b);
  }

  std::vector<phi::DenseTensor> input;
  input.push_back(input_a);
  input.push_back(input_b);

  phi::funcs::ConcatFunctor<DeviceContext, int> concat_functor;
  concat_functor(*context, input, 1, &out);

  // check the dim of input_a, input_b
  PADDLE_ENFORCE_EQ(input_a.dims(),
                    dim_a,
                    common::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_a.dims(),
                        dim_a));
  PADDLE_ENFORCE_EQ(input_b.dims(),
                    dim_b,
                    common::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_b.dims(),
                        dim_b));

  int* out_ptr = nullptr;
  if (phi::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(out, phi::CPUPlace(), &out_cpu);
    out_ptr = out_cpu.data<int>();
  } else {
    out_ptr = out.data<int>();
  }

  int cols = 3 * 4;
  int idx_a = 0, idx_b = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 28; ++j) {
      if (j >= cols) {
        PADDLE_ENFORCE_EQ(
            out_ptr[i * 28 + j],
            b_ptr[idx_b],
            common::errors::InvalidArgument(
                "Concat test failed, the result should be equal."));
        ++idx_b;
      } else {
        PADDLE_ENFORCE_EQ(
            out_ptr[i * 28 + j],
            a_ptr[idx_a],
            common::errors::InvalidArgument(
                "Concat test failed, the result should be equal."));
        ++idx_a;
      }
    }
  }
}

/**
 * case 3:
 *    inputs:
 *        t_a.shape: [2, 3, 5]
 *        t_b.shape: [2, 3, 4]
 *    output:
 *        out.shape: [2, 3, 9]
 */
template <typename DeviceContext, typename Place>
void ConcatCase3(DeviceContext* context) {
  phi::DenseTensor input_a_cpu;
  phi::DenseTensor input_b_cpu;
  phi::DenseTensor out_cpu;

  phi::DenseTensor input_a;
  phi::DenseTensor input_b;
  phi::DenseTensor out;

  auto dim_a = common::make_ddim({2, 3, 4});
  auto dim_b = common::make_ddim({2, 3, 5});
  auto dim_out = common::make_ddim({2, 3, 9});

  input_a.mutable_data<int>(dim_a, Place());
  input_b.mutable_data<int>(dim_b, Place());
  out.mutable_data<int>(dim_out, Place());

  if (phi::is_gpu_place(Place())) {
    input_a_cpu.mutable_data<int>(dim_a, phi::CPUPlace());
    input_b_cpu.mutable_data<int>(dim_b, phi::CPUPlace());
    out_cpu.mutable_data<int>(dim_out, phi::CPUPlace());
  }

  int* a_ptr = nullptr;
  int* b_ptr = nullptr;
  if (phi::is_gpu_place(Place())) {
    a_ptr = input_a_cpu.data<int>();
    b_ptr = input_b_cpu.data<int>();
  } else {
    a_ptr = input_a.data<int>();
    b_ptr = input_b.data<int>();
  }

  for (int i = 0; i < 2 * 3 * 4; ++i) {
    a_ptr[i] = i;
  }
  for (int i = 0; i < 2 * 3 * 5; ++i) {
    b_ptr[i] = i;
  }

  if (phi::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(input_a_cpu, Place(), &input_a);
    paddle::framework::TensorCopySync(input_b_cpu, Place(), &input_b);
  }

  std::vector<phi::DenseTensor> input;
  input.push_back(input_a);
  input.push_back(input_b);

  phi::funcs::ConcatFunctor<DeviceContext, int> concat_functor;
  concat_functor(*context, input, 2, &out);

  // check the dim of input_a, input_b
  PADDLE_ENFORCE_EQ(input_a.dims(),
                    dim_a,
                    common::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_a.dims(),
                        dim_a));
  PADDLE_ENFORCE_EQ(input_b.dims(),
                    dim_b,
                    common::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_b.dims(),
                        dim_b));

  int* out_ptr = nullptr;
  if (phi::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(out, phi::CPUPlace(), &out_cpu);
    out_ptr = out_cpu.data<int>();
  } else {
    out_ptr = out.data<int>();
  }

  // check the data
  int cols = 4;
  int idx_a = 0, idx_b = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 9; ++j) {
      if (j >= cols) {
        PADDLE_ENFORCE_EQ(
            out_ptr[i * 9 + j],
            b_ptr[idx_b],
            common::errors::InvalidArgument(
                "Concat test failed, the result should be equal."));
        ++idx_b;
      } else {
        PADDLE_ENFORCE_EQ(
            out_ptr[i * 9 + j],
            a_ptr[idx_a],
            common::errors::InvalidArgument(
                "Concat test failed, the result should be equal."));
        ++idx_a;
      }
    }
  }
}

/**
 * case 4:
 *    inputs:
 *        axis = 1
 *        t_a.shape: [2, 3, 4]
 *        t_b.shape: [2, 3, 4]
 *    output:
 *        out.shape: [2, 6, 4]
 */
template <typename DeviceContext, typename Place>
void ConcatCase4(DeviceContext* context) {
  phi::DenseTensor input_a_cpu;
  phi::DenseTensor input_b_cpu;
  phi::DenseTensor out_cpu;

  phi::DenseTensor input_a;
  phi::DenseTensor input_b;
  phi::DenseTensor out;

  auto dim_a = common::make_ddim({2, 3, 4});
  auto dim_b = common::make_ddim({2, 3, 4});
  auto dim_out = common::make_ddim({2, 6, 4});

  input_a.mutable_data<int>(dim_a, Place());
  input_b.mutable_data<int>(dim_b, Place());
  out.mutable_data<int>(dim_out, Place());

  if (phi::is_gpu_place(Place())) {
    input_a_cpu.mutable_data<int>(dim_a, phi::CPUPlace());
    input_b_cpu.mutable_data<int>(dim_b, phi::CPUPlace());
    out_cpu.mutable_data<int>(dim_out, phi::CPUPlace());
  }

  int* a_ptr = nullptr;
  int* b_ptr = nullptr;
  if (phi::is_gpu_place(Place())) {
    a_ptr = input_a_cpu.data<int>();
    b_ptr = input_b_cpu.data<int>();
  } else {
    a_ptr = input_a.data<int>();
    b_ptr = input_b.data<int>();
  }

  for (int i = 0; i < 2 * 3 * 4; ++i) {
    a_ptr[i] = i;
  }
  for (int i = 0; i < 2 * 3 * 4; ++i) {
    b_ptr[i] = i;
  }

  if (phi::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(input_a_cpu, Place(), &input_a);
    paddle::framework::TensorCopySync(input_b_cpu, Place(), &input_b);
  }

  std::vector<phi::DenseTensor> input;
  input.push_back(input_a);
  input.push_back(input_b);

  phi::funcs::ConcatFunctor<DeviceContext, int> concat_functor;
  concat_functor(*context, input, 1, &out);
  context->Wait();

  // check the dim of input_a, input_b
  PADDLE_ENFORCE_EQ(input_a.dims(),
                    dim_a,
                    common::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_a.dims(),
                        dim_a));
  PADDLE_ENFORCE_EQ(input_b.dims(),
                    dim_b,
                    common::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_b.dims(),
                        dim_b));

  int* out_ptr = nullptr;
  if (phi::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(out, phi::CPUPlace(), &out_cpu);
    out_ptr = out_cpu.data<int>();
  } else {
    out_ptr = out.data<int>();
  }

  // check the data
  int cols = 12;
  int idx_a = 0, idx_b = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 24; ++j) {
      if (j >= cols) {
        PADDLE_ENFORCE_EQ(
            out_ptr[i * 24 + j],
            b_ptr[idx_b],
            common::errors::InvalidArgument(
                "Concat test failed, the result should be equal."));
        ++idx_b;
      } else {
        PADDLE_ENFORCE_EQ(
            out_ptr[i * 24 + j],
            a_ptr[idx_a],
            common::errors::InvalidArgument(
                "Concat test failed, the result should be equal."));
        ++idx_a;
      }
    }
  }
}

template <typename DeviceContext, typename Place>
void TestConcatMain() {
  DeviceContext* context = new DeviceContext(Place());

  ConcatCase1<DeviceContext, Place>(context);
  ConcatCase2<DeviceContext, Place>(context);
  ConcatCase3<DeviceContext, Place>(context);
  ConcatCase4<DeviceContext, Place>(context);

  delete context;
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <>
void TestConcatMain<phi::GPUContext, phi::GPUPlace>() {
  auto* context = new phi::GPUContext(phi::GPUPlace());
  context->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                            .GetAllocator(phi::GPUPlace(), context->stream())
                            .get());
  context->PartialInitWithAllocator();

  ConcatCase1<phi::GPUContext, phi::GPUPlace>(context);
  ConcatCase2<phi::GPUContext, phi::GPUPlace>(context);
  ConcatCase3<phi::GPUContext, phi::GPUPlace>(context);
  ConcatCase4<phi::GPUContext, phi::GPUPlace>(context);

  delete context;
}
#endif

TEST(math, concat) {
  TestConcatMain<phi::CPUContext, phi::CPUPlace>();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  TestConcatMain<phi::GPUContext, phi::GPUPlace>();
#endif
}
