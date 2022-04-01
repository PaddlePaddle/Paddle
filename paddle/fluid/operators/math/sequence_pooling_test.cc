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

#include "paddle/fluid/operators/math/sequence_pooling.h"
#include <gtest/gtest.h>

template <typename DeviceContext, typename T>
void TestSequencePoolingSum(const DeviceContext &context,
                            const paddle::framework::LoD &lod,
                            const int64_t second_dim) {
  paddle::framework::LoDTensor cpu_out_grad;
  paddle::framework::LoDTensor cpu_in_grad;
  paddle::framework::LoDTensor out_grad;
  paddle::framework::LoDTensor in_grad;

  // construct out_grad's tensor in cpu
  const size_t out_first_dim = lod[0].size() - 1;
  auto out_dims =
      phi::make_ddim({static_cast<int64_t>(out_first_dim), second_dim});

  cpu_out_grad.mutable_data<T>(out_dims, paddle::platform::CPUPlace());
  for (int64_t i = 0; i < cpu_out_grad.numel(); ++i) {
    cpu_out_grad.data<T>()[i] = static_cast<T>(i);
  }

  // copy to dst out_grad
  auto place = context.GetPlace();
  if (paddle::platform::is_cpu_place(place)) {
    out_grad = cpu_out_grad;
  } else {
    paddle::framework::TensorCopySync(cpu_out_grad, place, &out_grad);
  }

  // construct in_grad
  in_grad.set_lod(lod);
  auto in_dims =
      phi::make_ddim({static_cast<int64_t>(lod[0].back()), second_dim});
  in_grad.mutable_data<T>(in_dims, place);

  // check tensor contruction result
  PADDLE_ENFORCE_EQ(
      in_grad.dims().size(), out_grad.dims().size(),
      paddle::platform::errors::InvalidArgument(
          "The dimension of input and output shall be same. Expected %ld == "
          "%ld, but got %ld != %ld. Please check the input value.",
          in_grad.dims().size(), out_grad.dims().size(), in_grad.dims().size(),
          out_grad.dims().size()));
  for (int64_t i = 1; i < out_grad.dims().size(); ++i) {
    PADDLE_ENFORCE_EQ(
        in_grad.dims()[i], out_grad.dims()[i],
        paddle::platform::errors::InvalidArgument(
            "The dimension of input and output shall be same. Expected %ld == "
            "%ld, but got %ld != %ld. Please check the input value.",
            in_grad.dims()[i], out_grad.dims()[i], in_grad.dims()[i],
            out_grad.dims()[i]));
  }

  // call functor
  paddle::operators::math::SequencePoolGradFunctor<DeviceContext, T>()(
      context, "SUM", out_grad, &in_grad);

  if (paddle::platform::is_cpu_place(place)) {
    cpu_in_grad = in_grad;
  } else {
    paddle::framework::TensorCopySync(in_grad, paddle::platform::CPUPlace(),
                                      &cpu_in_grad);
    cpu_in_grad.set_lod(in_grad.lod());
  }

  EXPECT_EQ(in_grad.numel(), static_cast<int64_t>(lod[0].back() * second_dim));
  EXPECT_EQ(in_grad.lod(), lod);

  if (paddle::platform::is_cpu_place(place)) {
    for (size_t i = 0; i < in_grad.lod()[0].size() - 1; ++i) {
      int64_t begin = in_grad.lod()[0][i];
      int64_t end = in_grad.lod()[0][i + 1];
      paddle::framework::Tensor tmp = in_grad.Slice(begin, end);
      for (int64_t j = 0; j != tmp.numel() / second_dim; ++j) {
        for (int64_t m = 0; m != second_dim; ++m) {
          EXPECT_EQ(tmp.data<T>()[m + j * second_dim],
                    out_grad.data<T>()[m + i * second_dim]);
        }
      }
    }
  } else {
    for (size_t i = 0; i < cpu_in_grad.lod()[0].size() - 1; ++i) {
      int64_t begin = cpu_in_grad.lod()[0][i];
      int64_t end = cpu_in_grad.lod()[0][i + 1];
      paddle::framework::Tensor tmp = cpu_in_grad.Slice(begin, end);
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
  auto place = paddle::platform::CPUPlace();
  auto *context = static_cast<paddle::platform::CPUDeviceContext *>(
      paddle::platform::DeviceContextPool::Instance().Get(place));

  paddle::framework::LoD lod1;
  lod1.push_back(std::vector<size_t>{0, 10});
  TestSequencePoolingSum<paddle::platform::CPUDeviceContext, float>(*context,
                                                                    lod1, 128);

  paddle::framework::LoD lod2;
  lod2.push_back(std::vector<size_t>{0, 2, 7, 10});
  TestSequencePoolingSum<paddle::platform::CPUDeviceContext, float>(*context,
                                                                    lod2, 128);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(SequencePoolingGrad, CUDA_SUM) {
  auto place = paddle::platform::CUDAPlace(0);
  auto *context = static_cast<paddle::platform::CUDADeviceContext *>(
      paddle::platform::DeviceContextPool::Instance().Get(place));

  paddle::framework::LoD lod1;
  lod1.push_back(std::vector<size_t>{0, 10});
  TestSequencePoolingSum<paddle::platform::CUDADeviceContext, float>(*context,
                                                                     lod1, 128);

  paddle::framework::LoD lod2;
  lod2.push_back(std::vector<size_t>{0, 2, 7, 10});
  TestSequencePoolingSum<paddle::platform::CUDADeviceContext, float>(*context,
                                                                     lod2, 128);
}
#endif
