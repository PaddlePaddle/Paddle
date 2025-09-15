// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/dlpack_tensor.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace framework {

template <typename T>
void TestMain(const phi::Place &place) {
  DDim dims{4, 5, 6, 7};
  phi::DenseTensor tensor;
  tensor.Resize(dims);
  void *p = tensor.mutable_data<T>(place);

  ::DLManagedTensor *dl_managed_tensor = paddle::framework::ToDLPack(tensor);
  ::DLTensor &dl_tensor = dl_managed_tensor->dl_tensor;

  PADDLE_ENFORCE_EQ(
      p,
      dl_tensor.data,
      common::errors::InvalidArgument("Tensor data pointer should be "
                                      "equal to DLPack "
                                      "tensor data pointer, but got "
                                      "tensor data pointer: %p, "
                                      "DLPack tensor data pointer: %p",
                                      p,
                                      dl_tensor.data));
  if (phi::is_cpu_place(place)) {
    PADDLE_ENFORCE_EQ(
        kDLCPU,
        dl_tensor.device.device_type,
        common::errors::InvalidArgument("Device type should be kDLCPU, "
                                        "but got %d",
                                        dl_tensor.device.device_type));
    PADDLE_ENFORCE_EQ(
        0,
        dl_tensor.device.device_id,
        common::errors::InvalidArgument("Device ID should be 0,"
                                        "but got %d",
                                        dl_tensor.device.device_id));
  } else if (phi::is_gpu_place(place)) {
    PADDLE_ENFORCE_EQ(kDLCUDA,
                      dl_tensor.device.device_type,
                      common::errors::InvalidArgument(
                          "Device type should be kDLCUDA, but got %d",
                          dl_tensor.device.device_type));
    PADDLE_ENFORCE_EQ(
        place.device,
        dl_tensor.device.device_id,
        common::errors::InvalidArgument("Device ID should be %d, "
                                        "but got %d",
                                        place.device,
                                        dl_tensor.device.device_id));
  } else if (phi::is_cuda_pinned_place(place)) {
    PADDLE_ENFORCE_EQ(
        kDLCUDAHost,
        dl_tensor.device.device_type,
        common::errors::InvalidArgument("Device type should be kDLCUDAHost, "
                                        "but got %d",
                                        dl_tensor.device.device_type));
    PADDLE_ENFORCE_EQ(
        0,
        dl_tensor.device.device_id,
        common::errors::InvalidArgument("Device ID should be 0, "
                                        "but got %d",
                                        dl_tensor.device.device_id));
  } else {
    PADDLE_ENFORCE_EQ(
        false, true, common::errors::InvalidArgument("Unsupported place type"));
  }

  PADDLE_ENFORCE_EQ(
      dims.size(),
      dl_tensor.ndim,
      common::errors::InvalidArgument("Dimension size should be equal to %d,"
                                      "but got %d",
                                      dims.size(),
                                      dl_tensor.ndim));
  for (auto i = 0; i < dims.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        dims[i],
        dl_tensor.shape[i],
        common::errors::InvalidArgument("Dimension at index %d should be %d, "
                                        "but got %d",
                                        i,
                                        dims[i],
                                        dl_tensor.shape[i]));
  }

  std::vector<int64_t> expect_strides(dims.size());
  expect_strides[dims.size() - 1] = 1;
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
    expect_strides[i] = expect_strides[i + 1] * dims[i + 1];
  }
  for (auto i = 0; i < dims.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        expect_strides[i],
        dl_tensor.strides[i],
        common::errors::InvalidArgument("Stride at index %d should be %d, "
                                        "but got %d",
                                        i,
                                        expect_strides[i],
                                        dl_tensor.strides[i]));
  }
  PADDLE_ENFORCE_EQ(static_cast<uint64_t>(0),
                    dl_tensor.byte_offset,
                    common::errors::InvalidArgument("Byte offset should be 0, "
                                                    "but got %d",
                                                    dl_tensor.byte_offset));

  PADDLE_ENFORCE_EQ(
      dl_tensor.dtype.lanes,
      1,
      common::errors::InvalidArgument(
          "Lanes should be %d, but got %d", 1, dl_tensor.dtype.lanes));
  PADDLE_ENFORCE_EQ(
      sizeof(T) * 8,
      dl_tensor.dtype.bits,
      common::errors::InvalidArgument("Data type bits should be %d, "
                                      "but got %d",
                                      sizeof(T) * 8,
                                      dl_tensor.dtype.bits));
}

template <typename T>
void TestToDLManagedTensor(const phi::Place &place) {
  DDim dims{6, 7};
  phi::DenseTensor tensor;
  tensor.Resize(dims);
  tensor.mutable_data<T>(place);

  ::DLManagedTensor *dl_managed_tensor = paddle::framework::ToDLPack(tensor);

  PADDLE_ENFORCE_NOT_NULL(
      dl_managed_tensor->manager_ctx,
      common::errors::InvalidArgument("Manager context should not be nullptr"));

  for (auto i = 0; i < dims.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        dims[i],
        dl_managed_tensor->dl_tensor.shape[i],
        common::errors::InvalidArgument("Dimension at index %d should be %d, "
                                        "but got %d",
                                        i,
                                        dims[i],
                                        dl_managed_tensor->dl_tensor.shape[i]));
  }

  PADDLE_ENFORCE_EQ(dl_managed_tensor->dl_tensor.strides[0] == 7,
                    true,
                    common::errors::InvalidArgument(
                        "Stride at index 0 should be 7, but got %d",
                        dl_managed_tensor->dl_tensor.strides[0]));
  PADDLE_ENFORCE_EQ(dl_managed_tensor->dl_tensor.strides[1] == 1,
                    true,
                    common::errors::InvalidArgument(
                        "Stride at index 1 should be 1, but got %d",
                        dl_managed_tensor->dl_tensor.strides[1]));

  dl_managed_tensor->deleter(dl_managed_tensor);
}

template <typename T>
void TestMainLoop() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::vector<phi::Place> places{
      phi::CPUPlace(), phi::GPUPlace(0), phi::GPUPinnedPlace()};
  if (platform::GetGPUDeviceCount() > 1) {
    places.emplace_back(phi::GPUPlace(1));
  }
#else
  std::vector<phi::Place> places{phi::CPUPlace()};
#endif
  for (auto &p : places) {
    TestMain<T>(p);
    TestToDLManagedTensor<T>(p);
  }
}
TEST(dlpack, test_all) {
#define TestCallback(cpp_type, proto_type) TestMainLoop<cpp_type>()

  _ForEachDataType_(TestCallback);
}

}  // namespace framework
}  // namespace paddle
