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

#include "paddle/phi/kernels/funcs/sequence_padding.h"

#include <gtest/gtest.h>
template <typename DeviceContext, typename T>
void TestSequencePadding(const DeviceContext &context,
                         const phi::LoD &lod,
                         const size_t sequence_width) {
  phi::DenseTensor cpu_seq;
  phi::DenseTensor cpu_seq_back;
  phi::DenseTensor seq;
  phi::DenseTensor seq_back;
  phi::DenseTensor padding;
  phi::DenseTensor cpu_pad_value;
  phi::DenseTensor pad_value;

  const size_t level = lod.size() - 1;
  auto seq_dims = phi::make_ddim({static_cast<int64_t>(lod[level].back()),
                                  static_cast<int64_t>(sequence_width)});

  cpu_seq.set_lod(lod);
  cpu_seq.mutable_data<T>(seq_dims, phi::CPUPlace());
  for (int64_t i = 0; i < cpu_seq.numel(); ++i) {
    cpu_seq.data<T>()[i] = static_cast<T>(i);
  }

  auto place = context.GetPlace();
  if (place.GetType() == phi::AllocationType::CPU) {
    seq = cpu_seq;
  } else {
    phi::Copy(context, cpu_seq, place, true, &seq);
    seq.set_lod(lod);
  }

  const size_t max_sequence_length =
      phi::funcs::MaximumSequenceLength(lod[level]);
  const size_t num_sequences = lod[level].size() - 1;
  auto padding_dims = phi::make_ddim({static_cast<int64_t>(max_sequence_length),
                                      static_cast<int64_t>(num_sequences),
                                      static_cast<int64_t>(sequence_width)});

  padding.mutable_data<T>(padding_dims, place);

  T *pad_value_data = cpu_pad_value.mutable_data<T>({1}, phi::CPUPlace());
  *pad_value_data = static_cast<T>(0);
  if (place.GetType() == phi::AllocationType::CPU) {
    pad_value = cpu_pad_value;
  } else {
    phi::Copy(context, cpu_pad_value, place, true, &pad_value);
  }

  phi::funcs::PaddingLoDTensorFunctor<DeviceContext, T>()(
      context,
      seq,
      &padding,
      pad_value,
      -1,
      0,
      false,
      paddle::operators::math::kLengthBatchWidth);

  seq_back.set_lod(lod);
  seq_back.mutable_data<T>(seq_dims, place);
  phi::funcs::UnpaddingLoDTensorFunctor<DeviceContext, T>()(
      context,
      padding,
      &seq_back,
      -1,
      0,
      false,
      paddle::operators::math::kLengthBatchWidth);

  if (place.GetType() == phi::AllocationType::CPU) {
    cpu_seq_back = seq_back;
  } else {
    phi::Copy(context, seq_back, phi::CPUPlace(), true, &cpu_seq_back);
    cpu_seq_back.set_lod(lod);
  }

  EXPECT_EQ(cpu_seq.numel(), cpu_seq_back.numel());
  EXPECT_EQ(cpu_seq.dims(), cpu_seq_back.dims());
  for (int64_t i = 0; i < cpu_seq.numel(); ++i) {
    EXPECT_EQ(cpu_seq.data<T>()[i], cpu_seq_back.data<T>()[i]);
  }
}

TEST(Seq2BatchPadding, CPU) {
  auto place = phi::CPUPlace();
  auto *context = static_cast<phi::CPUContext *>(
      paddle::platform::DeviceContextPool::Instance().Get(place));

  phi::LoD : lod1;
  lod1.push_back(std::vector<size_t>{0, 10});
  TestSequencePadding<phi::CPUContext, float>(*context, lod1, 16);

  phi::LoD : lod2;
  lod2.push_back(std::vector<size_t>{0, 2, 7, 10});
  TestSequencePadding<phi::CPUContext, float>(*context, lod2, 128);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(SequencePadding, CUDA) {
  auto place = paddle::platform::CUDAPlace(0);
  auto *context = static_cast<phi::GPUContext *>(
      paddle::platform::DeviceContextPool::Instance().Get(place));

  phi::LoD : lod1;
  lod1.push_back(std::vector<size_t>{0, 10});
  TestSequencePadding<phi::GPUContext, float>(*context, lod1, 16);

  phi::LoD : lod2;
  lod2.push_back(std::vector<size_t>{0, 2, 7, 10});
  TestSequencePadding<phi::GPUContext, float>(*context, lod2, 128);
}
#endif
