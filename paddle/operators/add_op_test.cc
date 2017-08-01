/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#define private public
#include <paddle/framework/op_registry.h>
USE_OP(add_two);
// USE_OP(add_two_grad);

TEST(AddOp, GetOpProto) {
  auto& protos = paddle::framework::OpRegistry::protos();
  auto it = protos.find("add_two");
  ASSERT_NE(it, protos.end());
  auto& op_creators = paddle::framework::OpRegistry::op_creators();
  auto it1 = op_creators.find("add_two_grad");
  ASSERT_NE(it1, op_creators.end());
}
#ifndef PADDLE_ONLY_CPU
TEST(AddOp, Kernel) {
  Tensor t;
  float* p = t.mutable_data<float>(make_ddim({6}), platform::CPUPlace());
  for (int i = 0; i < 6; i++) {
    p[i] = static_cast<float>(i);
  }

  Tensor t1;
  float* p1 = t1.mutable_data<float>(make_ddim({6}), platform::CPUPlace());
  for (int i = 0; i < 6; i++) {
    p1[i] = static_cast<float>(i);
  }

  Tensor t2;
  float* p2 = t2.mutable_data<float>(make_ddim({6}), platform::CPUPlace());
  for (int i = 0; i < 6; i++) {
    p2[i] = static_cast<float>(i);
  }

  Tensor t3;
  float* p3 = t3.mutable_data<float>(make_ddim({6}), platform::CPUPlace());
  for (int i = 0; i < 6; i++) {
    p3[i] = static_cast<float>(i);
  }

  t1.mutable_data<float>(platform::GPUPlace(0));
  t2.mutable_data<float>(platform::GPUPlace(0));

  t1.CopyFrom<T>(t, platform::GPUPlace(0));
  t2.CopyFrom<T>(t, platform::GPUPlace(0));

  t3.mutable_data<float>(platform::GPUPlace(0));

  CUDADeviceContext* dd = new CUDADeviceContext(0);

  EigenVector<T>::Flatten(t3).device(*(dd->eigen_device())) =
      EigenVector<T>::Flatten(t1) + EigenVector<T>::Flatten(t1);
}
#endif
