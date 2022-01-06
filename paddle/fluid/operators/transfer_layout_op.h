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

namespace paddle {
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace framework {
class LoDTensor;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
using DataLayout = framework::DataLayout;

class TransferLayoutFunctor {
 public:
  TransferLayoutFunctor(const framework::Variable *in, framework::Variable *out,
                        const platform::DeviceContext &dev_ctx,
                        const int dst_layout)
      : in_(in), out_(out), dev_ctx_(dev_ctx), dst_layout_(dst_layout) {}

  void operator()() const {
    auto &in_tensor = *framework::GetLoDTensorOrSelectedRowsValueFromVar(*in_);
    framework::LoDTensor out_tensor;

    auto out_layout = static_cast<DataLayout>(dst_layout_);
    out_tensor.set_layout(out_layout);

#ifdef PADDLE_WITH_MKLDNN
    auto in_layout = in_tensor.layout();
    if (in_layout == DataLayout::kMKLDNN || out_layout == DataLayout::kMKLDNN) {
      PADDLE_ENFORCE_NE(
          in_layout, out_layout,
          platform::errors::PreconditionNotMet(
              "No layout transform needed between two MKLDNN OPKernels."));

      if (in_layout != DataLayout::kMKLDNN &&
          out_layout == DataLayout::kMKLDNN) {
        // Case1 - transform from Non-MKLDNN OPKernel to MKLDNN OPKernel
        // Just set layout/format. No real transform occur

        auto out_format = platform::MKLDNNFormatForSize(
            in_tensor.dims().size(), framework::ToMKLDNNFormat(in_layout));
        out_tensor.ShareDataWith(in_tensor);
        // For NHWC data we need reshape of tensors as MKL-DNN
        // is expecting NHWC dims description order
        platform::MatchShapeToLayout(&out_tensor, in_layout, out_layout);
        paddle::platform::MKLDNNDeviceContext::tls().set_cur_paddle_data_layout(
            in_layout);
        out_tensor.set_layout(DataLayout::kMKLDNN);
        out_tensor.set_format(out_format);
      } else {
        // Case2 - transfrom from MKLDNN OPKernel to Non-MKLDNN OPKernel
        // Do transform via MKLDNN lib
        innerTransDataLayoutFromMKLDNN(
            in_layout, paddle::platform::MKLDNNDeviceContext::tls()
                           .get_cur_paddle_data_layout(),
            in_tensor, &out_tensor, dev_ctx_.GetPlace());
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
                       const framework::Tensor &in,
                       framework::Tensor *out) const {
    PADDLE_ENFORCE_EQ(
        framework::arity(in.dims()), 4,
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

    out->Resize(framework::make_ddim(dst_dim));
    out->mutable_data(in.place(), in.type());

    framework::VisitDataType(
        in.type(), framework::CastDataLayout(&dev_ctx, axis, in, out));
  }

  const framework::Variable *in_;
  framework::Variable *out_;
  const platform::DeviceContext &dev_ctx_;
  const int dst_layout_;
};

}  // namespace operators
}  // namespace paddle
