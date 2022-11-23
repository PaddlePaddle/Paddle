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

#include "paddle/fluid/framework/data_transform.h"

#include "paddle/fluid/framework/data_device_transform.h"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/data_type_transform.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace framework {

static void PassTensorData(phi::DenseTensor *from, phi::DenseTensor *to) {
  to->ShareDataWith(*from);
  *from = phi::DenseTensor();
}

void TransformData(const OpKernelType &expected_kernel_type,
                   const OpKernelType &kernel_type_for_var,
                   const phi::DenseTensor &input_tensor,
                   phi::DenseTensor *output_tensor) {
  bool transformed = false;
  phi::DenseTensor in;
  in.ShareDataWith(input_tensor);
  phi::DenseTensor out;
  const DataLayout lin = kernel_type_for_var.data_layout_;
  const DataLayout lout = expected_kernel_type.data_layout_;
  // do layout transform
  if (NeedTransformLayout(lout, lin)) {
#ifdef PADDLE_WITH_MKLDNN
    if (lin == DataLayout::kMKLDNN || lout == DataLayout::kMKLDNN) {
      PADDLE_ENFORCE_EQ(
          !(lin == DataLayout::kMKLDNN && lout == DataLayout::kMKLDNN),
          true,
          platform::errors::PreconditionNotMet(
              "No layout transform needed between two MKLDNN OPKernels."));

      if (lin != DataLayout::kMKLDNN && lout == DataLayout::kMKLDNN) {
        // Case1 - transform from Non-MKLDNN OPKernel to MKLDNN OPKernel
        // Just set layout/format. No real transform occur

        auto out_format = platform::MKLDNNFormatForSize(in.dims().size(),
                                                        ToMKLDNNFormat(lin));
        out.ShareDataWith(input_tensor);
        // For NHWC data we need reshape of tensors as MKL-DNN
        // is expecting NHWC dims description order
        if (lin == DataLayout::kNHWC || lin == DataLayout::kNDHWC) {
          platform::MatchShapeToLayout(&out, lin, lout);
          // We register only NHWC assuming that model is consistent e.g. either
          // NHWC or NCHW
          paddle::platform::MKLDNNDeviceContext::tls()
              .set_cur_paddle_data_layout(lin);
        }
        dnnl::memory::desc out_mem_desc(
            vectorize(out.dims()),
            ToMKLDNNDataType(TransToProtoVarType(in.type())),
            out_format);
        out.set_mem_desc(out_mem_desc);
      } else {
        // Case2 - transfrom from MKLDNN OPKernel to Non-MKLDNN OPKernel
        // Do transform via MKLDNN lib
        TransDataLayoutFromMKLDNN(
            kernel_type_for_var, expected_kernel_type, in, &out);
      }
    } else {
      // Case3 - transfrom between Non-MKLDNN OPKernels
      TransDataLayout(kernel_type_for_var, expected_kernel_type, in, &out);
    }
#else
    // Case3 - transfrom between Non-MKLDNN OPKernels
    TransDataLayout(kernel_type_for_var, expected_kernel_type, in, &out);
#endif
    transformed = true;
    PassTensorData(&out, &in);
  }

  // do data type transform
  if (expected_kernel_type.data_type_ != kernel_type_for_var.data_type_) {
    TransDataType(kernel_type_for_var, expected_kernel_type, in, &out);
    transformed = true;
    PassTensorData(&out, &in);
  }

  // do device transform
  if (!platform::is_same_place(kernel_type_for_var.place_,
                               expected_kernel_type.place_)) {
    TransDataDevice(in, expected_kernel_type.place_, &out);
    transformed = true;
    PassTensorData(&out, &in);
  }

  PADDLE_ENFORCE_EQ(
      transformed,
      true,
      platform::errors::PreconditionNotMet(
          "No transform is applied for the data needs to be transformed."));
  // get output data
  output_tensor->ShareDataWith(in);
}

void SetTensorToVariable(const Variable &in_var,
                         const phi::DenseTensor &tensor,
                         Variable *out_var) {
  if (in_var.IsType<LoDTensor>()) {
    auto &in_lod_tensor = in_var.Get<LoDTensor>();
    auto *tran_lod_tensor = out_var->GetMutable<LoDTensor>();
    tran_lod_tensor->set_lod(in_lod_tensor.lod());
    tran_lod_tensor->set_layout(in_lod_tensor.layout());
#ifdef PADDLE_WITH_MKLDNN
    tran_lod_tensor->set_mem_desc(in_lod_tensor.mem_desc());
#endif
    tran_lod_tensor->ShareDataWith(tensor);
  } else if (in_var.IsType<phi::SelectedRows>()) {
    auto &in_selected_rows = in_var.Get<phi::SelectedRows>();
    auto *trans_selected_rows = out_var->GetMutable<phi::SelectedRows>();
    trans_selected_rows->set_height(in_selected_rows.height());
    trans_selected_rows->set_rows(in_selected_rows.rows());
    trans_selected_rows->mutable_value()->ShareDataWith(tensor);
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported variable type, only supports LoDTensor or SelectedRows, "
        "but the input variable type is %s.",
        ToTypeName(in_var.Type())));
  }
}

}  // namespace framework
}  // namespace paddle
