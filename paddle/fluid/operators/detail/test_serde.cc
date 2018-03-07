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
#include <thread>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/printf.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace operators = paddle::operators;

TEST(Tensor, CPU) {
  // serialize var to ByteBuffer
  framework::Variable var;
  auto* tensor = var.GetMutable<framework::LoDTensor>();
  tensor->Resize(framework::make_ddim({4, 8, 4, 2}));
  int tensor_numel = 4 * 8 * 4 * 2;
  platform::CPUPlace place;
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& ctx = *pool.Get(place);
  float* orig_tensor_data = tensor->mutable_data<float>(place);
  for (int i = 0; i < tensor_numel; ++i) orig_tensor_data[i] = i;

  ::grpc::ByteBuffer msg;
  operators::detail::SerializeToByteBuffer("myvar", &var, ctx, &msg);
  EXPECT_GT(msg.Length(), 0);

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
  EXPECT_EQ(varmsg.dims()[0], 4);
  EXPECT_EQ(varmsg.dims()[1], 8);
  EXPECT_EQ(varmsg.dims()[2], 4);
  EXPECT_EQ(varmsg.dims()[3], 2);
  std::cout << "payload size " << varmsg.serialized().size() << std::endl;
  const float* tensor_data =
      reinterpret_cast<const float*>(varmsg.serialized().data());
  for (int i = 0; i < tensor_numel; ++i)
    EXPECT_EQ(tensor_data[i], orig_tensor_data[i]);

  // deserialize zero-copy
  framework::Variable var2;
  operators::detail::DeserializeFromByteBuffer(msg, ctx, &var2);
  auto tensor2 = var2.Get<framework::LoDTensor>();
  const float* tensor_data2 = tensor2.data<float>();
  for (int i = 0; i < tensor_numel; ++i)
    EXPECT_EQ(tensor_data2[i], orig_tensor_data[i]);
}