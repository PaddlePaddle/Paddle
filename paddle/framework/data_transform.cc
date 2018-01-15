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

#include "paddle/framework/data_transform.h"

#include "paddle/framework/data_device_transform.h"
#include "paddle/framework/data_layout_transform.h"

namespace paddle {
namespace framework {

void DataTransform(const OpKernelType& expected_kernel_type,
                   const OpKernelType& kernel_type_for_var,
                   const Tensor& input_tensor, Tensor* out) {
  std::shared_ptr<const Tensor> input_ptr(&input_tensor);
  std::shared_ptr<Tensor> output_ptr(new Tensor());

  // do layout transform
  std::shared_ptr<Tensor> layout_transform_out(new Tensor());
  if (expected_kernel_type.data_layout_ != kernel_type_for_var.data_layout_) {
    TransDataLayout(kernel_type_for_var, expected_kernel_type, *input_ptr,
                    layout_transform_out.get());
    input_ptr = layout_transform_out;
    output_ptr = layout_transform_out;
  }

  // do device transform
  std::shared_ptr<Tensor> device_transform_out(new Tensor());
  if (!platform::is_same_place(kernel_type_for_var.place_,
                               expected_kernel_type.place_)) {
    DeviceTransform(*input_ptr, expected_kernel_type.place_,
                    device_transform_out.get());
    input_ptr = layout_transform_out;
    output_ptr = layout_transform_out;
  }

  out->ShareDataWith(*output_ptr);
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

}  // namespace framework
}  // namespace paddle
