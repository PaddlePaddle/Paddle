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
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

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
  paddle::framework::Tensor input_a_cpu;
  paddle::framework::Tensor input_b_cpu;
  paddle::framework::Tensor out_cpu;

  paddle::framework::Tensor input_a;
  paddle::framework::Tensor input_b;
  paddle::framework::Tensor out;

  auto dim_a = phi::make_ddim({2, 3, 4});
  auto dim_b = phi::make_ddim({3, 3, 4});
  auto dim_out = phi::make_ddim({5, 3, 4});

  input_a.mutable_data<int>(dim_a, Place());
  input_b.mutable_data<int>(dim_b, Place());
  out.mutable_data<int>(dim_out, Place());

  if (paddle::platform::is_gpu_place(Place())) {
    input_a_cpu.mutable_data<int>(dim_a, paddle::platform::CPUPlace());
    input_b_cpu.mutable_data<int>(dim_b, paddle::platform::CPUPlace());
    out_cpu.mutable_data<int>(dim_out, paddle::platform::CPUPlace());
  }

  int* a_ptr = nullptr;
  int* b_ptr = nullptr;
  if (paddle::platform::is_gpu_place(Place())) {
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

  if (paddle::platform::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(input_a_cpu, Place(), &input_a);
    paddle::framework::TensorCopySync(input_b_cpu, Place(), &input_b);
  }

  std::vector<paddle::framework::Tensor> input;
  input.push_back(input_a);
  input.push_back(input_b);

  paddle::operators::math::ConcatFunctor<DeviceContext, int> concat_functor;
  concat_functor(*context, input, 0, &out);

  // check the dim of input_a, input_b
  PADDLE_ENFORCE_EQ(input_a.dims(), dim_a,
                    paddle::platform::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_a.dims(), dim_a));
  PADDLE_ENFORCE_EQ(input_b.dims(), dim_b,
                    paddle::platform::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_b.dims(), dim_b));

  int* out_ptr = nullptr;
  if (paddle::platform::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(out, paddle::platform::CPUPlace(),
                                      &out_cpu);
    out_ptr = out_cpu.data<int>();
  } else {
    out_ptr = out.data<int>();
  }

  int cols = 2 * 3 * 4;
  int idx_a = 0, idx_b = 0;
  for (int j = 0; j < 5 * 3 * 4; ++j) {
    if (j >= cols) {
      PADDLE_ENFORCE_EQ(out_ptr[j], b_ptr[idx_b],
                        paddle::platform::errors::InvalidArgument(
                            "Concat test failed, the result should be equal."));
      ++idx_b;
    } else {
      PADDLE_ENFORCE_EQ(out_ptr[j], a_ptr[idx_a],
                        paddle::platform::errors::InvalidArgument(
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
  paddle::framework::Tensor input_a_cpu;
  paddle::framework::Tensor input_b_cpu;
  paddle::framework::Tensor out_cpu;

  paddle::framework::Tensor input_a;
  paddle::framework::Tensor input_b;
  paddle::framework::Tensor out;

  auto dim_a = phi::make_ddim({2, 3, 4});
  auto dim_b = phi::make_ddim({2, 4, 4});
  auto dim_out = phi::make_ddim({2, 7, 4});

  input_a.mutable_data<int>(dim_a, Place());
  input_b.mutable_data<int>(dim_b, Place());
  out.mutable_data<int>(dim_out, Place());

  if (paddle::platform::is_gpu_place(Place())) {
    input_a_cpu.mutable_data<int>(dim_a, paddle::platform::CPUPlace());
    input_b_cpu.mutable_data<int>(dim_b, paddle::platform::CPUPlace());
    out_cpu.mutable_data<int>(dim_out, paddle::platform::CPUPlace());
  }

  int* a_ptr = nullptr;
  int* b_ptr = nullptr;
  if (paddle::platform::is_gpu_place(Place())) {
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

  if (paddle::platform::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(input_a_cpu, Place(), &input_a);
    paddle::framework::TensorCopySync(input_b_cpu, Place(), &input_b);
  }

  std::vector<paddle::framework::Tensor> input;
  input.push_back(input_a);
  input.push_back(input_b);

  paddle::operators::math::ConcatFunctor<DeviceContext, int> concat_functor;
  concat_functor(*context, input, 1, &out);

  // check the dim of input_a, input_b
  PADDLE_ENFORCE_EQ(input_a.dims(), dim_a,
                    paddle::platform::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_a.dims(), dim_a));
  PADDLE_ENFORCE_EQ(input_b.dims(), dim_b,
                    paddle::platform::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_b.dims(), dim_b));

  int* out_ptr = nullptr;
  if (paddle::platform::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(out, paddle::platform::CPUPlace(),
                                      &out_cpu);
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
            out_ptr[i * 28 + j], b_ptr[idx_b],
            paddle::platform::errors::InvalidArgument(
                "Concat test failed, the result should be equal."));
        ++idx_b;
      } else {
        PADDLE_ENFORCE_EQ(
            out_ptr[i * 28 + j], a_ptr[idx_a],
            paddle::platform::errors::InvalidArgument(
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
  paddle::framework::Tensor input_a_cpu;
  paddle::framework::Tensor input_b_cpu;
  paddle::framework::Tensor out_cpu;

  paddle::framework::Tensor input_a;
  paddle::framework::Tensor input_b;
  paddle::framework::Tensor out;

  auto dim_a = phi::make_ddim({2, 3, 4});
  auto dim_b = phi::make_ddim({2, 3, 5});
  auto dim_out = phi::make_ddim({2, 3, 9});

  input_a.mutable_data<int>(dim_a, Place());
  input_b.mutable_data<int>(dim_b, Place());
  out.mutable_data<int>(dim_out, Place());

  if (paddle::platform::is_gpu_place(Place())) {
    input_a_cpu.mutable_data<int>(dim_a, paddle::platform::CPUPlace());
    input_b_cpu.mutable_data<int>(dim_b, paddle::platform::CPUPlace());
    out_cpu.mutable_data<int>(dim_out, paddle::platform::CPUPlace());
  }

  int* a_ptr = nullptr;
  int* b_ptr = nullptr;
  if (paddle::platform::is_gpu_place(Place())) {
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

  if (paddle::platform::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(input_a_cpu, Place(), &input_a);
    paddle::framework::TensorCopySync(input_b_cpu, Place(), &input_b);
  }

  std::vector<paddle::framework::Tensor> input;
  input.push_back(input_a);
  input.push_back(input_b);

  paddle::operators::math::ConcatFunctor<DeviceContext, int> concat_functor;
  concat_functor(*context, input, 2, &out);

  // check the dim of input_a, input_b
  PADDLE_ENFORCE_EQ(input_a.dims(), dim_a,
                    paddle::platform::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_a.dims(), dim_a));
  PADDLE_ENFORCE_EQ(input_b.dims(), dim_b,
                    paddle::platform::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_b.dims(), dim_b));

  int* out_ptr = nullptr;
  if (paddle::platform::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(out, paddle::platform::CPUPlace(),
                                      &out_cpu);
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
            out_ptr[i * 9 + j], b_ptr[idx_b],
            paddle::platform::errors::InvalidArgument(
                "Concat test failed, the result should be equal."));
        ++idx_b;
      } else {
        PADDLE_ENFORCE_EQ(
            out_ptr[i * 9 + j], a_ptr[idx_a],
            paddle::platform::errors::InvalidArgument(
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
  paddle::framework::Tensor input_a_cpu;
  paddle::framework::Tensor input_b_cpu;
  paddle::framework::Tensor out_cpu;

  paddle::framework::Tensor input_a;
  paddle::framework::Tensor input_b;
  paddle::framework::Tensor out;

  auto dim_a = phi::make_ddim({2, 3, 4});
  auto dim_b = phi::make_ddim({2, 3, 4});
  auto dim_out = phi::make_ddim({2, 6, 4});

  input_a.mutable_data<int>(dim_a, Place());
  input_b.mutable_data<int>(dim_b, Place());
  out.mutable_data<int>(dim_out, Place());

  if (paddle::platform::is_gpu_place(Place())) {
    input_a_cpu.mutable_data<int>(dim_a, paddle::platform::CPUPlace());
    input_b_cpu.mutable_data<int>(dim_b, paddle::platform::CPUPlace());
    out_cpu.mutable_data<int>(dim_out, paddle::platform::CPUPlace());
  }

  int* a_ptr = nullptr;
  int* b_ptr = nullptr;
  if (paddle::platform::is_gpu_place(Place())) {
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

  if (paddle::platform::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(input_a_cpu, Place(), &input_a);
    paddle::framework::TensorCopySync(input_b_cpu, Place(), &input_b);
  }

  std::vector<paddle::framework::Tensor> input;
  input.push_back(input_a);
  input.push_back(input_b);

  paddle::operators::math::ConcatFunctor<DeviceContext, int> concat_functor;
  concat_functor(*context, input, 1, &out);
  context->Wait();

  // check the dim of input_a, input_b
  PADDLE_ENFORCE_EQ(input_a.dims(), dim_a,
                    paddle::platform::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_a.dims(), dim_a));
  PADDLE_ENFORCE_EQ(input_b.dims(), dim_b,
                    paddle::platform::errors::InvalidArgument(
                        "The dims of Input tensor should be the same as the "
                        "declared dims. Tensor dims: [%s], declared dims: [%s]",
                        input_b.dims(), dim_b));

  int* out_ptr = nullptr;
  if (paddle::platform::is_gpu_place(Place())) {
    paddle::framework::TensorCopySync(out, paddle::platform::CPUPlace(),
                                      &out_cpu);
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
            out_ptr[i * 24 + j], b_ptr[idx_b],
            paddle::platform::errors::InvalidArgument(
                "Concat test failed, the result should be equal."));
        ++idx_b;
      } else {
        PADDLE_ENFORCE_EQ(
            out_ptr[i * 24 + j], a_ptr[idx_a],
            paddle::platform::errors::InvalidArgument(
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
void TestConcatMain<paddle::platform::CUDADeviceContext,
                    paddle::platform::CUDAPlace>() {
  auto* context =
      new paddle::platform::CUDADeviceContext(paddle::platform::CUDAPlace());
  context->SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CUDAPlace(), context->stream())
          .get());
  context->PartialInitWithAllocator();

  ConcatCase1<paddle::platform::CUDADeviceContext, paddle::platform::CUDAPlace>(
      context);
  ConcatCase2<paddle::platform::CUDADeviceContext, paddle::platform::CUDAPlace>(
      context);
  ConcatCase3<paddle::platform::CUDADeviceContext, paddle::platform::CUDAPlace>(
      context);
  ConcatCase4<paddle::platform::CUDADeviceContext, paddle::platform::CUDAPlace>(
      context);

  delete context;
}
#endif

TEST(math, concat) {
  TestConcatMain<paddle::platform::CPUDeviceContext,
                 paddle::platform::CPUPlace>();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  TestConcatMain<paddle::platform::CUDADeviceContext,
                 paddle::platform::CUDAPlace>();
#endif
}
