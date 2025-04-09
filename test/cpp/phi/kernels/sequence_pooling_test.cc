/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/sequence_pooling.h"

template <typename DeviceContext, typename T>
void TestSequencePoolingSum(const DeviceContext &context,
                            const phi::LegacyLoD &lod,
                            const int64_t second_dim) {
  phi::DenseTensor cpu_out_grad;
  phi::DenseTensor cpu_in_grad;
  phi::DenseTensor out_grad;
  phi::DenseTensor in_grad;

  // construct out_grad's tensor in cpu
  const size_t out_first_dim = lod[0].size() - 1;
  auto out_dims =
      common::make_ddim({static_cast<int64_t>(out_first_dim), second_dim});

  cpu_out_grad.mutable_data<T>(out_dims, phi::CPUPlace());
  for (int64_t i = 0; i < cpu_out_grad.numel(); ++i) {
    cpu_out_grad.data<T>()[i] = static_cast<T>(i);
  }

  // copy to dst out_grad
  auto place = context.GetPlace();
  if (place == phi::CPUPlace()) {
    out_grad = cpu_out_grad;
  } else {
    phi::Copy(context, cpu_out_grad, place, true, &out_grad);
  }

  // construct in_grad
  in_grad.set_lod(lod);
  auto in_dims =
      common::make_ddim({static_cast<int64_t>(lod[0].back()), second_dim});
  in_grad.mutable_data<T>(in_dims, place);

  // check tensor construction result
  PADDLE_ENFORCE_EQ(
      in_grad.dims().size(),
      out_grad.dims().size(),
      common::errors::InvalidArgument(
          "The dimension of input and output shall be same. Expected %ld == "
          "%ld, but got %ld != %ld. Please check the input value.",
          in_grad.dims().size(),
          out_grad.dims().size(),
          in_grad.dims().size(),
          out_grad.dims().size()));
  for (int64_t i = 1; i < out_grad.dims().size(); ++i) {
    PADDLE_ENFORCE_EQ(
        in_grad.dims()[i],
        out_grad.dims()[i],
        common::errors::InvalidArgument(
            "The dimension of input and output shall be same. Expected %ld == "
            "%ld, but got %ld != %ld. Please check the input value.",
            in_grad.dims()[i],
            out_grad.dims()[i],
            in_grad.dims()[i],
            out_grad.dims()[i]));
  }

  // call functor
  phi::funcs::SequencePoolGradFunctor<DeviceContext, T>()(
      context, "SUM", out_grad, &in_grad);

  if (place == phi::CPUPlace()) {
    cpu_in_grad = in_grad;
  } else {
    phi::Copy(context, in_grad, phi::CPUPlace(), true, &cpu_in_grad);
    cpu_in_grad.set_lod(in_grad.lod());
  }

  EXPECT_EQ(in_grad.numel(), static_cast<int64_t>(lod[0].back() * second_dim));
  EXPECT_EQ(in_grad.lod(), lod);

  if (place == phi::CPUPlace()) {
    for (size_t i = 0; i < in_grad.lod()[0].size() - 1; ++i) {
      int64_t begin = static_cast<int64_t>(in_grad.lod()[0][i]);
      int64_t end = static_cast<int64_t>(in_grad.lod()[0][i + 1]);
      phi::DenseTensor tmp = in_grad.Slice(begin, end);
      for (int64_t j = 0; j != tmp.numel() / second_dim; ++j) {
        for (int64_t m = 0; m != second_dim; ++m) {
          EXPECT_EQ(tmp.data<T>()[m + j * second_dim],
                    out_grad.data<T>()[m + i * second_dim]);
        }
      }
    }
  } else {
    for (size_t i = 0; i < cpu_in_grad.lod()[0].size() - 1; ++i) {
      int64_t begin = static_cast<int64_t>(cpu_in_grad.lod()[0][i]);
      int64_t end = static_cast<int64_t>(cpu_in_grad.lod()[0][i + 1]);
      phi::DenseTensor tmp = cpu_in_grad.Slice(begin, end);
      for (int64_t j = 0; j != tmp.numel() / second_dim; ++j) {
        for (int64_t m = 0; m != second_dim; ++m) {
          EXPECT_EQ(tmp.data<T>()[m + j * second_dim],
                    cpu_out_grad.data<T>()[m + i * second_dim]);
        }
      }
    }
  }
}

TEST(SequencePoolingGrad, CPU_SUM) {
  auto place = phi::CPUPlace();
  auto *context = static_cast<phi::CPUContext *>(
      phi::DeviceContextPool::Instance().Get(place));

  phi::LegacyLoD lod1;
  lod1.push_back(std::vector<size_t>{0, 10});
  TestSequencePoolingSum<phi::CPUContext, float>(*context, lod1, 128);

  phi::LegacyLoD lod2;
  lod2.push_back(std::vector<size_t>{0, 2, 7, 10});
  TestSequencePoolingSum<phi::CPUContext, float>(*context, lod2, 128);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(SequencePoolingGrad, CUDA_SUM) {
  auto place = phi::GPUPlace(0);
  auto *context = static_cast<phi::GPUContext *>(
      phi::DeviceContextPool::Instance().Get(place));

  phi::LegacyLoD lod1;
  lod1.push_back(std::vector<size_t>{0, 10});
  TestSequencePoolingSum<phi::GPUContext, float>(*context, lod1, 128);

  phi::LegacyLoD lod2;
  lod2.push_back(std::vector<size_t>{0, 2, 7, 10});
  TestSequencePoolingSum<phi::GPUContext, float>(*context, lod2, 128);
}
#endif
