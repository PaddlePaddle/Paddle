/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/transfer_layout_kernel.h"

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

std::vector<int> GetAxis(const DataLayout& from, const DataLayout& to) {
  PADDLE_ENFORCE_NE(
      from,
      to,
      common::errors::InvalidArgument(
          "Layout transform should transform between different layout."));
  if (from == DataLayout::NCHW && to == DataLayout::NHWC) {
    return {0, 2, 3, 1};
  } else if (from == DataLayout::NHWC && to == DataLayout::NCHW) {
    return {0, 3, 1, 2};
  } else {
    PADDLE_THROW(
        common::errors::InvalidArgument("Unsupported layout transform."));
  }
}

template <typename T, typename Context>
void CastDataLayout(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<int>& axis,
                    DenseTensor* out) {
  funcs::Transpose<Context, T, 4> trans4;
  trans4(dev_ctx, x, out, axis);
}

template <typename Context>
void TransferLayoutGeneral(const Context& dev_ctx,
                           const DenseTensor& x,
                           DataLayout dst_layout,
                           DenseTensor* out) {
  auto src_dim = x.dims();

  auto axis = GetAxis(x.layout(), dst_layout);

  std::vector<int64_t> dst_dim;
  dst_dim.resize(axis.size());
  for (size_t i = 0; i < axis.size(); i++) {
    dst_dim[i] = src_dim[axis[i]];
  }

  out->Resize(common::make_ddim(dst_dim));
  dev_ctx.Alloc(out, x.dtype());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // In GPU fp16 model, we will insert many transfer_layout ops in
  // transfer_layout_pass, so we optimize this kernel on GPU
  if (std::is_same<Context, phi::GPUContext>::value) {
    std::vector<int> axis_nchw_nhwc = {0, 2, 3, 1};
    std::vector<int> axis_nhwc_nchw = {0, 3, 1, 2};
    auto* gpu_ctx = reinterpret_cast<const phi::GPUContext*>(&dev_ctx);
    const int64_t batch = src_dim[0];
    int64_t row_len = src_dim[1];
    int64_t col_len = src_dim[2] * src_dim[3];
    if (axis == axis_nhwc_nchw) {
      row_len = src_dim[1] * src_dim[2];
      col_len = src_dim[3];
    }
    if (x.dtype() == phi::DataType::FLOAT16) {
      funcs::BatchTranspose(out->data<phi::dtype::float16>(),
                            x.data<phi::dtype::float16>(),
                            batch,
                            row_len,
                            col_len,
                            gpu_ctx);
      return;
    } else if (x.dtype() == phi::DataType::FLOAT32) {
      funcs::BatchTranspose(out->data<float>(),
                            x.data<float>(),
                            batch,
                            row_len,
                            col_len,
                            gpu_ctx);
      return;
    } else if (x.dtype() == phi::DataType::BFLOAT16) {
      funcs::BatchTranspose(out->data<phi::dtype::bfloat16>(),
                            x.data<phi::dtype::bfloat16>(),
                            batch,
                            row_len,
                            col_len,
                            gpu_ctx);
      return;
    }
  }
#endif

  PD_VISIT_ALL_TYPES(x.dtype(), "CastDataLayout", ([&] {
                       CastDataLayout<data_t, Context>(dev_ctx, x, axis, out);
                     }));
}

#ifdef PADDLE_WITH_DNNL
template <typename Context>
void TransferLayoutMKLDNN(const Context& dev_ctx,
                          const DenseTensor& x,
                          DataLayout src_layout,
                          DataLayout dst_layout,
                          DenseTensor* out) {
  auto print_tensor_meta = [](const DenseTensor& x) {
    std::ostringstream oss;

    oss << "[";
    oss << "layout:" << x.layout() << " ,";
    oss << "dims:" << x.dims() << " ,";
    if (x.IsInitialized()) oss << "place:" << x.place();
    oss << "]";

    return oss.str();
  };
  VLOG(10) << " x: " << print_tensor_meta(x);
  VLOG(10) << " out: " << print_tensor_meta(*out) << " " << out;

  // NOTE(zhiqiu): to handle the special case in ApplyDataTransform() in
  // data_transfer.cc
  if (!x.IsInitialized() && src_layout == DataLayout::ONEDNN &&
      dst_layout == DataLayout::NHWC) {
    VLOG(4) << src_layout << "->" << dst_layout << " " << x.layout();
    out->Resize(x.dims());
    out->set_layout(dst_layout);
    funcs::MatchShapeToLayout(out, src_layout, dst_layout);
    return;
  }

  if (src_layout != DataLayout::ONEDNN && dst_layout == DataLayout::ONEDNN) {
    // Case1 - transform from Non-MKLDNN OPKernel to OneDNN OPKernel
    // Just set layout/format. No real transform occur
    out->ShareDataWith(x);
    // For NHWC data we need reshape of tensors as MKL-DNN
    // is expecting NHWC dims description order
    if (src_layout == DataLayout::NHWC) {
      VLOG(4) << "NHWC";
      funcs::MatchShapeToLayout(out, src_layout, dst_layout);
      OneDNNContext::tls().set_cur_paddle_data_layout(src_layout);
    }

    dnnl::memory::desc out_mem_desc = funcs::make_memory_desc(*out, src_layout);
    out->set_mem_desc(out_mem_desc);
  } else if (src_layout == DataLayout::ONEDNN &&
             dst_layout != DataLayout::ONEDNN) {
    // Case2 - transform from OneDNN OPKernel to Non-MKLDNN OPKernel
    // Do transform via OneDNN lib
    funcs::TransDataLayoutFromOneDNN(
        src_layout, dst_layout, x, out, dev_ctx.GetPlace());
  } else if (src_layout == DataLayout::ONEDNN &&
             dst_layout == DataLayout::ONEDNN) {
    PADDLE_ENFORCE_NE(
        src_layout,
        dst_layout,
        errors::PreconditionNotMet(
            "No layout transform needed between two oneDNN OPKernels."));
  } else {
    TransferLayoutGeneral<Context>(dev_ctx, x, dst_layout, out);
  }
}
#endif

template <typename Context>
void TransferLayoutKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          int src_layout,
                          int dst_layout,
                          DenseTensor* out) {
  PADDLE_ENFORCE_NE(src_layout,
                    dst_layout,
                    errors::PreconditionNotMet(
                        "No layout transform needed between same layout."));
  VLOG(10) << "TransDataLayout from " << static_cast<DataLayout>(src_layout)
           << " -> " << static_cast<DataLayout>(dst_layout);

  VLOG_IF(10, x.initialized()) << "TransDataLayout from " << x.layout();
  if (x.layout() == static_cast<DataLayout>(dst_layout)) {
    VLOG(10) << "No need to transform, already is " << x.layout();
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

#ifdef PADDLE_WITH_DNNL
  TransferLayoutMKLDNN<Context>(dev_ctx,
                                x,
                                static_cast<DataLayout>(src_layout),
                                static_cast<DataLayout>(dst_layout),
                                out);
#else
  TransferLayoutGeneral<Context>(
      dev_ctx, x, static_cast<DataLayout>(dst_layout), out);
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_DTYPE(transfer_layout,
                                 CPU,
                                 ALL_LAYOUT,
                                 phi::TransferLayoutKernel<phi::CPUContext>) {}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL_FOR_ALL_DTYPE(transfer_layout,
                                 GPU,
                                 ALL_LAYOUT,
                                 phi::TransferLayoutKernel<phi::GPUContext>) {}
#endif
