/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <unistd.h>
#include <string>
#include <thread>  // NOLINT

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"
#include "paddle/fluid/operators/distributed/variable_response.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/printf.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace operators = paddle::operators;
namespace math = paddle::operators::math;
namespace memory = paddle::memory;

void RunSerdeTestSelectedRows(platform::Place place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& ctx = *pool.Get(place);

  // serialize var to ByteBuffer
  framework::Variable var;
  auto* slr = var.GetMutable<framework::SelectedRows>();
  slr->set_height(1000);
  auto* tensor = slr->mutable_value();
  auto* rows = slr->mutable_rows();
  tensor->Resize(framework::make_ddim({564, 128}));
  tensor->mutable_data<float>(place);
  int tensor_numel = 564 * 128;
  math::set_constant(ctx, tensor, 32.7);
  for (int i = 0; i < 564; ++i) rows->push_back(i);

  ::grpc::ByteBuffer msg;
  operators::distributed::SerializeToByteBuffer("myvar", &var, ctx, &msg);
  EXPECT_GT(msg.Length(), static_cast<size_t>(0));

  // deserialize
  std::vector<::grpc::Slice> slices;
  (void)msg.Dump(&slices);
  std::string tmp;
  for (const auto& s : slices) {
    tmp.append(reinterpret_cast<const char*>(s.begin()), s.size());
  }

  sendrecv::VariableMessage varmsg;
  EXPECT_TRUE(varmsg.ParseFromString(tmp));

  // deserialize bytebuffer
  EXPECT_EQ(varmsg.varname(), "myvar");
  EXPECT_EQ(varmsg.type(), 1);

  const float* tensor_data =
      reinterpret_cast<const float*>(varmsg.serialized().data());
  const int64_t* rows_data =
      reinterpret_cast<const int64_t*>(varmsg.rows().data());
  for (int i = 0; i < tensor_numel; ++i) {
    EXPECT_FLOAT_EQ(tensor_data[i], 32.7);
  }
  for (int i = 0; i < 564; ++i) {
    EXPECT_EQ(rows_data[i], i);
  }

  // deserialize zero-copy
  // framework::Variable var2;
  // operators::distributed::DeserializeFromByteBuffer(msg, ctx, &var2);
  framework::Scope scope;
  scope.Var("myvar");
  operators::distributed::VariableResponse resp(&scope, &ctx);
  EXPECT_EQ(resp.Parse(msg), 0);

  framework::Variable* var2 = resp.GetVar();

  auto* slr2 = var2->GetMutable<framework::SelectedRows>();
  auto* tensor2 = slr2->mutable_value();
  auto* rows2 = slr2->mutable_rows();
  float* tensor_data2 = nullptr;
  framework::Tensor tmp_tensor;

  if (platform::is_gpu_place(ctx.GetPlace())) {
    platform::CPUPlace cpu;
    framework::TensorCopy(*tensor2, cpu, &tmp_tensor);
    tensor_data2 = tmp_tensor.data<float>();
  } else {
    tensor_data2 = const_cast<float*>(tensor2->data<float>());
  }
  const int64_t* rows_data2 = rows2->data();

  for (int i = 0; i < tensor_numel; ++i) {
    EXPECT_FLOAT_EQ(tensor_data2[i], 32.7);
  }
  for (size_t i = 0; i < rows2->size(); ++i) {
    EXPECT_EQ(rows_data2[i], static_cast<int64_t>(i));
  }
  EXPECT_EQ(slr2->height(), 1000);
}

void RunTestLodTensor(platform::Place place, int from_type = 0) {
  // serialize var to ByteBuffer
  framework::Variable var;
  auto* tensor = var.GetMutable<framework::LoDTensor>();
  tensor->Resize(framework::make_ddim({512, 8, 4, 2}));
  framework::LoD lod;
  lod.push_back(framework::Vector<size_t>({1, 3, 8}));
  tensor->set_lod(lod);
  int tensor_numel = 512 * 8 * 4 * 2;
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& ctx = *pool.Get(place);
  tensor->mutable_data<float>(place);
  math::set_constant(ctx, tensor, 31.9);

  ::grpc::ByteBuffer msg;
  operators::distributed::SerializeToByteBuffer("myvar", &var, ctx, &msg);
  EXPECT_GT(msg.Length(), static_cast<size_t>(0));

  // deserialize
  std::vector<::grpc::Slice> slices;
  (void)msg.Dump(&slices);
  std::string tmp;
  for (const auto& s : slices) {
    tmp.append(reinterpret_cast<const char*>(s.begin()), s.size());
  }
  sendrecv::VariableMessage varmsg;
  EXPECT_TRUE(varmsg.ParseFromString(tmp));
  EXPECT_EQ(varmsg.varname(), "myvar");
  EXPECT_EQ(varmsg.type(), 0);
  EXPECT_EQ(varmsg.dims()[0], 512);
  EXPECT_EQ(varmsg.dims()[1], 8);
  EXPECT_EQ(varmsg.dims()[2], 4);
  EXPECT_EQ(varmsg.dims()[3], 2);
  EXPECT_EQ(varmsg.lod_level(), 1);
  EXPECT_EQ(varmsg.lod(0).lod_data(0), 1);
  EXPECT_EQ(varmsg.lod(0).lod_data(1), 3);
  EXPECT_EQ(varmsg.lod(0).lod_data(2), 8);

  const float* tensor_data =
      reinterpret_cast<const float*>(varmsg.serialized().data());
  for (int i = 0; i < tensor_numel; ++i) {
    EXPECT_FLOAT_EQ(tensor_data[i], 31.9);
  }

  // message binary
  std::string str;
  varmsg.SerializeToString(&str);

  // message bytebuffer
  ::grpc::Slice slices_2[1];
  int num_slices = 1;
  slices_2[0] = ::grpc::Slice(str.length());
  memcpy(const_cast<uint8_t*>(slices_2[0].begin()), str.c_str(), str.length());
  ::grpc::ByteBuffer bytebuffer2(&slices_2[0], num_slices);

  // deserialize zero-copy
  framework::Scope scope;
  scope.Var("myvar");
  operators::distributed::VariableResponse resp(&scope, &ctx);
  if (from_type == 0) {
    EXPECT_EQ(resp.Parse(msg), 0);
  } else {
    EXPECT_EQ(resp.Parse(bytebuffer2), 0);
  }

  framework::Variable* var2 = resp.GetVar();

  auto tensor2 = var2->Get<framework::LoDTensor>();
  float* tensor_data2 = nullptr;
  framework::Tensor tmp_tensor;

  if (platform::is_gpu_place(ctx.GetPlace())) {
    platform::CPUPlace cpu;
    framework::TensorCopy(tensor2, cpu, &tmp_tensor);
    tensor_data2 = tmp_tensor.data<float>();
  } else {
    tensor_data2 = const_cast<float*>(tensor2.data<float>());
  }

  EXPECT_EQ(varmsg.lod_level(), 1);
  EXPECT_EQ(varmsg.lod(0).lod_data(0), 1);
  EXPECT_EQ(varmsg.lod(0).lod_data(1), 3);
  EXPECT_EQ(varmsg.lod(0).lod_data(2), 8);
  for (int i = 0; i < tensor_numel; ++i) EXPECT_FLOAT_EQ(tensor_data2[i], 31.9);
}

TEST(LodTensor, Run) {
  platform::CPUPlace place;
  RunTestLodTensor(place);
  RunTestLodTensor(place, 1);
#ifdef PADDLE_WITH_CUDA
  platform::CUDAPlace gpu(0);
  RunTestLodTensor(gpu);
  RunTestLodTensor(gpu, 1);
#endif
}

TEST(SelectedRows, Run) {
  platform::CPUPlace place;
  RunSerdeTestSelectedRows(place);

#ifdef PADDLE_WITH_CUDA
  platform::CUDAPlace gpu;
  RunSerdeTestSelectedRows(gpu);
#endif
}
