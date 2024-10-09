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

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace framework {

namespace {  // NOLINT
template <typename T>
constexpr uint8_t GetDLDataTypeCode() {
  if (std::is_same<T, phi::dtype::complex<float>>::value ||
      std::is_same<T, phi::dtype::complex<double>>::value) {
    return static_cast<uint8_t>(kDLComplex);
  }

  if (std::is_same<T, phi::dtype::bfloat16>::value) {
    return static_cast<uint8_t>(kDLBfloat);
  }
  if (std::is_same<T, bool>::value) {
    return static_cast<uint8_t>(kDLBool);
  }

  return std::is_same<phi::dtype::float16, T>::value ||
                 std::is_floating_point<T>::value
             ? static_cast<uint8_t>(kDLFloat)
             : (std::is_unsigned<T>::value
                    ? static_cast<uint8_t>(kDLUInt)
                    : (std::is_integral<T>::value ? static_cast<uint8_t>(kDLInt)
                                                  : static_cast<uint8_t>(-1)));
}
}  // namespace

template <typename T>
void TestMain(const phi::Place &place, uint16_t lanes) {
  DDim dims{4, 5, 6, 7};
  phi::DenseTensor tensor;
  tensor.Resize(dims);
  void *p = tensor.mutable_data<T>(place);

  DLPackTensor dlpack_tensor(tensor, lanes);
  ::DLTensor &dl_tensor = dlpack_tensor;

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

  PADDLE_ENFORCE_EQ(
      dl_tensor.strides == nullptr,
      true,
      common::errors::InvalidArgument("Strides should be nullptr, "
                                      "but got non-nullptr value"));
  PADDLE_ENFORCE_EQ(static_cast<uint64_t>(0),
                    dl_tensor.byte_offset,
                    common::errors::InvalidArgument("Byte offset should be 0, "
                                                    "but got %d",
                                                    dl_tensor.byte_offset));

  PADDLE_ENFORCE_EQ(
      lanes,
      dl_tensor.dtype.lanes,
      common::errors::InvalidArgument(
          "Lanes should be %d, but got %d", lanes, dl_tensor.dtype.lanes));
  PADDLE_ENFORCE_EQ(
      sizeof(T) * 8,
      dl_tensor.dtype.bits,
      common::errors::InvalidArgument("Data type bits should be %d, "
                                      "but got %d",
                                      sizeof(T) * 8,
                                      dl_tensor.dtype.bits));

  PADDLE_ENFORCE_EQ(
      GetDLDataTypeCode<T>(),
      dl_tensor.dtype.code,
      common::errors::InvalidArgument("Data type code should be %d,"
                                      "but got %d",
                                      GetDLDataTypeCode<T>(),
                                      dl_tensor.dtype.code));
}

template <typename T>
void TestToDLManagedTensor(const phi::Place &place, uint16_t lanes) {
  DDim dims{6, 7};
  phi::DenseTensor tensor;
  tensor.Resize(dims);
  tensor.mutable_data<T>(place);

  DLPackTensor dlpack_tensor(tensor, lanes);

  ::DLManagedTensor *dl_managed_tensor = dlpack_tensor.ToDLManagedTensor();

  PADDLE_ENFORCE_EQ(
      dl_managed_tensor->manager_ctx == nullptr,
      true,
      common::errors::InvalidArgument("Manager context should be nullptr, "
                                      "but got non-nullptr value"));

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
  std::vector<uint16_t> lanes{1, 2};
  for (auto &p : places) {
    for (auto &l : lanes) {
      TestMain<T>(p, l);
      TestToDLManagedTensor<T>(p, l);
    }
  }
}
TEST(dlpack, test_all) {
#define TestCallback(cpp_type, proto_type) TestMainLoop<cpp_type>()

  _ForEachDataType_(TestCallback);
}

}  // namespace framework
}  // namespace paddle
