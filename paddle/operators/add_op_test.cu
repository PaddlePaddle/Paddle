#include "gtest/gtest.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"



TEST(AddOpKernel, Kernel) {
  paddle::framework::Tensor t;
  float* p = t.mutable_data<float>(paddle::framework::make_ddim({6}), paddle::platform::CPUPlace());
  for (int i = 0; i < 6; i++) {
    p[i] = static_cast<float>(i);
  }

  paddle::framework::Tensor t1;
  float* p1 = t1.mutable_data<float>(paddle::framework::make_ddim({6}), paddle::platform::CPUPlace());
  for (int i = 0; i < 6; i++) {
    p1[i] = static_cast<float>(i);
  }

  paddle::framework::Tensor t2;
  float* p2 = t2.mutable_data<float>(paddle::framework::make_ddim({6}), paddle::platform::CPUPlace());
  for (int i = 0; i < 6; i++) {
    p2[i] = static_cast<float>(i);
  }

  paddle::framework::Tensor t3;
  float* p3 = t3.mutable_data<float>(paddle::framework::make_ddim({6}), paddle::platform::CPUPlace());
  for (int i = 0; i < 6; i++) {
    p3[i] = static_cast<float>(i);
  }

  t1.mutable_data<float>(paddle::platform::GPUPlace(0));
  t2.mutable_data<float>(paddle::platform::GPUPlace(0));

  t1.CopyFrom<float>(t, paddle::platform::GPUPlace(0));
  t2.CopyFrom<float>(t, paddle::platform::GPUPlace(0));

  t3.mutable_data<float>(paddle::platform::GPUPlace(0));

  paddle::platform::CUDADeviceContext* dd =
      new paddle::platform::CUDADeviceContext(0);

  paddle::framework::EigenVector<float>::Flatten(t3).device(*(dd->eigen_device())) =
      paddle::framework::EigenVector<float>::Flatten(t1) +
      paddle::framework::EigenVector<float>::Flatten(t1);
}
