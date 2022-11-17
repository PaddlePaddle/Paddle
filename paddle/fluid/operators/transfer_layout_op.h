// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/device_context.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
using DataLayout = phi::DataLayout;

class TransferLayoutFunctor {
 public:
  TransferLayoutFunctor(const framework::Variable *in,
                        framework::Variable *out,
                        const platform::DeviceContext &dev_ctx,
                        const int src_layout,
                        const int dst_layout,
                        std::string in_name)
      : in_(in),
        out_(out),
        dev_ctx_(dev_ctx),
        src_layout_(src_layout),
        dst_layout_(dst_layout),
        in_name_(in_name) {}

  void operator()() const {
    auto &in_tensor = *framework::GetLoDTensorOrSelectedRowsValueFromVar(*in_);
    phi::DenseTensor out_tensor;

    auto out_layout = static_cast<DataLayout>(dst_layout_);
    out_tensor.set_layout(out_layout);

#ifdef PADDLE_WITH_MKLDNN
    // NOTE(zhiqiu): to handle the special case in ApplyDataTransform() in
    // data_transfer.cc
    auto in_layout = static_cast<DataLayout>(src_layout_);
    auto *tensor_out = out_->GetMutable<phi::DenseTensor>();
    VLOG(4) << in_layout << "->" << out_layout << " " << in_tensor.layout();
    if (!in_tensor.IsInitialized() && in_layout == DataLayout::ONEDNN &&
        out_layout == DataLayout::kNHWC) {
      tensor_out->Resize(in_tensor.dims());
      tensor_out->set_layout(out_layout);
      phi::funcs::MatchShapeToLayout(tensor_out, in_layout, out_layout);
      return;
    }
    if (in_layout == DataLayout::ONEDNN || out_layout == DataLayout::ONEDNN) {
      PADDLE_ENFORCE_NE(
          in_layout,
          out_layout,
          platform::errors::PreconditionNotMet(
              "No layout transform needed between two oneDNN OPKernels."));

      if (in_layout != DataLayout::ONEDNN && out_layout == DataLayout::ONEDNN) {
        // Case1 - transform from Non-MKLDNN OPKernel to MKLDNN OPKernel
        // Just set layout/format. No real transform occur

        auto out_format = phi::funcs::OneDNNFormatForSize(
            in_tensor.dims().size(), framework::ToOneDNNFormat(in_layout));
        out_tensor.ShareDataWith(in_tensor);
        // For NHWC data we need reshape of tensors as MKL-DNN
        // is expecting NHWC dims description order
        if (in_layout == DataLayout::kNHWC) {
          VLOG(4) << "kNHWC";
          phi::funcs::MatchShapeToLayout(&out_tensor, in_layout, out_layout);
          paddle::platform::MKLDNNDeviceContext::tls()
              .set_cur_paddle_data_layout(in_layout);
        }

        auto out_tz = phi::vectorize<int64_t>(out_tensor.dims());
        dnnl::memory::data_type in_type = framework::ToMKLDNNDataType(
            framework::TransToProtoVarType(in_tensor.dtype()));

        dnnl::memory::desc out_mem_desc(out_tz, in_type, out_format);
        out_tensor.set_mem_desc(out_mem_desc);
      } else {
        auto target_layout = paddle::platform::MKLDNNDeviceContext::tls()
                                 .get_cur_paddle_data_layout();
        // NOTE(zhiqiu): hot fix, follow the same logic in DataCopy() in
        // fetch_op.cc
        if (out_layout == DataLayout::kNCHW &&
            in_name_ == framework::GradVarName("Filter")) {
          target_layout = out_layout;
        }
        VLOG(4) << "innerTransDataLayoutFromMKLDNN: " << in_layout << "->"
                << target_layout;
        // Case2 - transfrom from MKLDNN OPKernel to Non-MKLDNN OPKernel
        // Do transform via MKLDNN lib
        paddle::framework::innerTransDataLayoutFromMKLDNN(in_layout,
                                                          target_layout,
                                                          in_tensor,
                                                          &out_tensor,
                                                          dev_ctx_.GetPlace());
      }
    } else {
      // Case3 - transfrom between Non-MKLDNN OPKernels
      TransDataLayout(dev_ctx_, in_tensor, &out_tensor);
    }
#else
    // Case3 - transfrom between Non-MKLDNN OPKernels
    TransDataLayout(dev_ctx_, in_tensor, &out_tensor);
#endif
    framework::SetTensorToVariable(*in_, out_tensor, out_);
  }

 private:
  void TransDataLayout(const platform::DeviceContext &dev_ctx,
                       const phi::DenseTensor &in,
                       phi::DenseTensor *out) const {
    PADDLE_ENFORCE_EQ(
        phi::arity(in.dims()),
        4,
        platform::errors::InvalidArgument(
            "Input dimension arity only can be 4, the input dimension is %s.",
            in.dims()));

    auto src_dim = in.dims();
    std::vector<int64_t> dst_dim;

    auto axis = framework::GetAxis(in.layout(), out->layout());
    dst_dim.resize(axis.size());
    for (size_t i = 0; i < axis.size(); i++) {
      dst_dim[i] = src_dim[axis[i]];
    }

    out->Resize(phi::make_ddim(dst_dim));
    out->mutable_data(in.place(), in.type());

    framework::VisitDataType(
        framework::TransToProtoVarType(in.dtype()),
        framework::CastDataLayout(&dev_ctx, axis, in, out));
  }

  const framework::Variable *in_;
  framework::Variable *out_;
  const platform::DeviceContext &dev_ctx_;
  const int src_layout_;
  const int dst_layout_;
  std::string in_name_;
};

}  // namespace operators
}  // namespace paddle
