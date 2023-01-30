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

<<<<<<< HEAD
static void PassTensorData(phi::DenseTensor *from, phi::DenseTensor *to) {
  to->ShareDataWith(*from);
  *from = phi::DenseTensor();
}

void TransformData(const phi::KernelKey &expected_kernel_type,
                   const phi::KernelKey &kernel_type_for_var,
                   const phi::DenseTensor &input_tensor,
                   phi::DenseTensor *output_tensor,
                   const phi::Place &place) {
  bool transformed = false;
  phi::DenseTensor in;
  in.ShareDataWith(input_tensor);
  phi::DenseTensor out;
  const DataLayout lin = kernel_type_for_var.layout();
  const DataLayout lout = expected_kernel_type.layout();
  // do layout transform
  if (NeedTransformLayout(lout, lin)) {
#ifdef PADDLE_WITH_MKLDNN
    if (lin == DataLayout::ONEDNN || lout == DataLayout::ONEDNN) {
      PADDLE_ENFORCE_EQ(
          !(lin == DataLayout::ONEDNN && lout == DataLayout::ONEDNN),
          true,
          platform::errors::PreconditionNotMet(
              "No layout transform needed between two oneDNN OPKernels."));

      if (lin != DataLayout::ONEDNN && lout == DataLayout::ONEDNN) {
        // Case1 - transform from Non-ONEDNN OPKernel to ONEDNN OPKernel
        // Just set layout/format. No real transform occur

        auto out_format = phi::funcs::OneDNNFormatForSize(
            in.dims().size(), phi::funcs::ToOneDNNFormat(lin));
=======
static void PassTensorData(Tensor *from, Tensor *to) {
  to->ShareDataWith(*from);
  *from = Tensor();
}

void TransformData(const OpKernelType &expected_kernel_type,
                   const OpKernelType &kernel_type_for_var,
                   const Tensor &input_tensor,
                   Tensor *output_tensor) {
  bool transformed = false;
  Tensor in;
  in.ShareDataWith(input_tensor);
  Tensor out;
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        out.ShareDataWith(input_tensor);
        // For NHWC data we need reshape of tensors as MKL-DNN
        // is expecting NHWC dims description order
        if (lin == DataLayout::kNHWC || lin == DataLayout::kNDHWC) {
<<<<<<< HEAD
          phi::funcs::MatchShapeToLayout(&out, lin, lout);
          // We register only NHWC assuming that model is consistent e.g. either
          // NHWC or NCHW
          phi::OneDNNContext::tls().set_cur_paddle_data_layout(lin);
        }
        dnnl::memory::desc out_mem_desc(
            vectorize(out.dims()),
            phi::funcs::ToOneDNNDataType(in.dtype()),
            out_format);
        out.set_mem_desc(out_mem_desc);
      } else {
        // Case2 - transfrom from ONEDNN OPKernel to Non-ONEDNN OPKernel
        // Do transform via ONEDNN lib
        PADDLE_ENFORCE(lin == DataLayout::ONEDNN && lout != DataLayout::ONEDNN,
                       platform::errors::InvalidArgument(
                           "TransDataLayoutFromOneDNN only supports "
                           "transform from ONEDNN to non-ONEDNN"));

        phi::funcs::TransDataLayoutFromOneDNN(
            lin,
            phi::OneDNNContext::tls().get_cur_paddle_data_layout(),
            in,
            &out,
            place);
      }
    } else {
      // Case3 - transfrom between Non-ONEDNN OPKernels
      TransDataLayout(
          kernel_type_for_var, expected_kernel_type, in, &out, place);
    }
#else
    // Case3 - transfrom between Non-ONEDNN OPKernels
    TransDataLayout(kernel_type_for_var, expected_kernel_type, in, &out, place);
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#endif
    transformed = true;
    PassTensorData(&out, &in);
  }

  // do data type transform
<<<<<<< HEAD
  if (NeedTransformDataType(expected_kernel_type, kernel_type_for_var)) {
=======
  if (expected_kernel_type.data_type_ != kernel_type_for_var.data_type_) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    TransDataType(kernel_type_for_var, expected_kernel_type, in, &out);
    transformed = true;
    PassTensorData(&out, &in);
  }

  // do device transform
<<<<<<< HEAD
  if (kernel_type_for_var.backend() != phi::Backend::ALL_BACKEND &&
      !platform::is_same_place(in.place(), place)) {
    TransDataDevice(in, place, &out);
=======
  if (!platform::is_same_place(kernel_type_for_var.place_,
                               expected_kernel_type.place_)) {
    TransDataDevice(in, expected_kernel_type.place_, &out);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
                         const phi::DenseTensor &tensor,
                         Variable *out_var) {
  if (in_var.IsType<phi::DenseTensor>()) {
    auto &in_lod_tensor = in_var.Get<phi::DenseTensor>();
    auto *tran_lod_tensor = out_var->GetMutable<phi::DenseTensor>();
=======
                         const Tensor &tensor,
                         Variable *out_var) {
  if (in_var.IsType<LoDTensor>()) {
    auto &in_lod_tensor = in_var.Get<LoDTensor>();
    auto *tran_lod_tensor = out_var->GetMutable<LoDTensor>();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        "Unsupported variable type, only supports phi::DenseTensor or "
        "SelectedRows, "
=======
        "Unsupported variable type, only supports LoDTensor or SelectedRows, "
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        "but the input variable type is %s.",
        ToTypeName(in_var.Type())));
  }
}

}  // namespace framework
}  // namespace paddle
