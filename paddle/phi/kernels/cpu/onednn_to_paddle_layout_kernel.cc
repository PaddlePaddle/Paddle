/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/onednn_to_paddle_layout_kernel.h"

#include <sstream>
#include <string>

#include "glog/logging.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/memcpy_kernel.h"
#ifdef PADDLE_WITH_DNNL
#include "paddle/phi/backends/onednn/onednn_helper.h"
#endif
namespace phi {

template <typename Context>
void OneDNN2PaddleLayout(const Context& dev_ctx,
                         const DenseTensor& x,
                         int dst_layout,
                         DenseTensor* out) {
#ifdef PADDLE_WITH_DNNL
  DataLayout src_layout = x.layout();
  VLOG(10) << "TransDataLayout from " << static_cast<DataLayout>(src_layout)
           << " -> " << static_cast<DataLayout>(dst_layout);

  auto print_tensor_meta = [](const DenseTensor& x) {
    std::ostringstream oss;

    oss << "[";
    oss << "layout:" << x.layout() << " ,";
    oss << "dims:" << x.dims() << " ,";
    oss << "dtype:" << x.dtype() << " ,";
    if (x.IsInitialized()) oss << "place:" << x.place();
    oss << "]";

    return oss.str();
  };
  VLOG(10) << " x: " << print_tensor_meta(x);
  VLOG(10) << " out: " << print_tensor_meta(*out) << " " << out;

  DataLayout tmp_layout = static_cast<DataLayout>(dst_layout);

  if (tmp_layout == DataLayout::ANY) {
    tmp_layout = phi::OneDNNContext::tls().get_cur_paddle_data_layout();
  }

  VLOG(4) << "src_layout: " << src_layout << ", tmp_layout: " << tmp_layout;

  if (src_layout != DataLayout::ONEDNN || !x.storage_properties_initialized()) {
    if (!x.IsInitialized()) {
      out->Resize(x.dims());
      out->set_layout(tmp_layout);
      return;
    }
    out->ShareDataWith(x);
    out->ShareInplaceVersionCounterWith(x);
    out->set_layout(static_cast<DataLayout>(tmp_layout));
    return;
  }

  // NOTE(zhiqiu): to handle the special case in ApplyDataTransform() in
  // data_transfer.cc
  if (!x.IsInitialized() && src_layout == DataLayout::ONEDNN &&
      tmp_layout == DataLayout::NHWC) {
    VLOG(4) << src_layout << "->" << tmp_layout << " " << x.layout();
    out->Resize(x.dims());
    out->set_layout(tmp_layout);
    funcs::MatchShapeToLayout(out, src_layout, tmp_layout);
    return;
  }

  funcs::TransDataLayoutFromOneDNN(
      src_layout, tmp_layout, x, out, dev_ctx.GetPlace());
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_DTYPE(onednn_to_paddle_layout,
                                 CPU,
                                 ALL_LAYOUT,
                                 phi::OneDNN2PaddleLayout<phi::CPUContext>) {}
