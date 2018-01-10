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
#include <functional>

#include "paddle/framework/data_layout_transform.h"
#include "paddle/framework/data_transform.h"
#include "paddle/framework/data_type_transform.h"
#include "paddle/framework/device_data_transform.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/selected_rows.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {

DataTransformFnMap& DataTransformFnMap::Instance() {
  static DataTransformFnMap data_transform_map;
  return data_transform_map;
}

Tensor* DataTransform(const OpKernelType& expected_kernel_type,
                      const OpKernelType& kernel_type_for_var,
                      const Tensor& input_tensor) {
  Tensor* out = nullptr;
  if (!platform::is_same_place(kernel_type_for_var.place_,
                               expected_kernel_type.place_)) {
    out = DeviceTransform(input_tensor, expected_kernel_type.place_);
  }
  PADDLE_ENFORCE_NOT_NULL(out, "out should not be null");
  return out;
}

void CopyVariableWithTensor(const Variable& in_var, const Tensor& tensor,
                            Variable& out_var) {
  if (in_var.IsType<LoDTensor>()) {
    auto& in_lod_tensor = in_var.Get<LoDTensor>();
    auto* tran_lod_tensor = out_var.GetMutable<LoDTensor>();
    tran_lod_tensor->set_lod(in_lod_tensor.lod());
    tran_lod_tensor->set_layout(in_lod_tensor.layout());
    tran_lod_tensor->ShareDataWith(tensor);
  } else if (in_var.IsType<SelectedRows>()) {
    auto& in_selected_rows = in_var.Get<SelectedRows>();
    auto* trans_selected_rows = out_var.GetMutable<SelectedRows>();
    trans_selected_rows->set_height(in_selected_rows.height());
    trans_selected_rows->set_rows(in_selected_rows.rows());
    trans_selected_rows->mutable_value()->ShareDataWith(tensor);
  } else {
    PADDLE_THROW("unknown var type");
  }
}

auto KernelFP32 = OpKernelType(proto::DataType::FP32, platform::CPUPlace(),
                               DataLayout::kNHWC, LibraryType::kPlain);

auto KernelFP64 = OpKernelType(proto::DataType::FP64, platform::CPUPlace(),
                               DataLayout::kNHWC, LibraryType::kPlain);

auto KernelNHWC = OpKernelType(proto::DataType::FP64, platform::CPUPlace(),
                               DataLayout::kNHWC, LibraryType::kPlain);

auto KernelNCHW = OpKernelType(proto::DataType::FP64, platform::CPUPlace(),
                               DataLayout::kNCHW, LibraryType::kPlain);

// TODO(dzhwinter): Only for testing multiple op kernel.
// Dummy transform function for library_type
// should be removed.
auto KernelPlain = OpKernelType(proto::DataType::FP32, platform::CUDAPlace(0),
                                DataLayout::kAnyLayout, LibraryType::kPlain);

auto KernelCUDNN = OpKernelType(proto::DataType::FP32, platform::CUDAPlace(0),
                                DataLayout::kAnyLayout, LibraryType::kCUDNN);

}  // namespace framework
}  // namespace paddle

namespace f = paddle::framework;

namespace {
std::vector<int> NHWC2NCHW = {0, 3, 1, 2};
std::vector<int> NCHW2NHWC = {0, 2, 3, 1};
}

REGISTER_DATA_TRANSFORM_FN(f::KernelFP32, f::KernelFP64, f::TransDataType);
REGISTER_DATA_TRANSFORM_FN(f::KernelPlain, f::KernelCUDNN, f::DummyTrans);
REGISTER_DATA_TRANSFORM_FN(f::KernelCUDNN, f::KernelPlain, f::DummyTrans);
REGISTER_DATA_TRANSFORM_FN(f::KernelNHWC, f::KernelNCHW,
                           std::bind(f::TransDataLayout, NHWC2NCHW,
                                     std::placeholders::_1,
                                     std::placeholders::_2,
                                     std::placeholders::_3,
                                     std::placeholders::_4));
REGISTER_DATA_TRANSFORM_FN(f::KernelNCHW, f::KernelNHWC,
                           std::bind(f::TransDataLayout, NCHW2NHWC,
                                     std::placeholders::_1,
                                     std::placeholders::_2,
                                     std::placeholders::_3,
                                     std::placeholders::_4));
