// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/framework/data_type_transform.h"

#include "paddle/phi/common/transform.h"
#include "paddle/phi/core/framework/convert_utils.h"
#include "paddle/phi/core/framework/selected_rows_serialize.h"
#include "paddle/phi/core/kernel_factory.h"

#if defined(PADDLE_WITH_XPU)
#include "paddle/phi/core/platform/device/device_wrapper.h"
#endif

namespace proto = paddle::framework::proto;

namespace phi {

template <typename InType, typename OutType>
struct CastDataTypeFunctor {
  HOSTDEVICE inline OutType operator()(InType in) const {
    return static_cast<OutType>(in);
  }
};

#if defined(PADDLE_WITH_XPU)

template <typename InType, typename OutType>
static void XPUCastData(const phi::DenseTensor& in,
                        phi::DenseTensor* out,
                        const phi::XPUContext* dev_ctx) {
  using XPUInTDType = typename XPUTypeTrait<InType>::Type;
  using XPUOutTDType = typename XPUTypeTrait<OutType>::Type;
  int r = xpu::cast<XPUInTDType, XPUOutTDType>(
      dev_ctx->x_context(),
      reinterpret_cast<const XPUInTDType*>(in.data<InType>()),
      reinterpret_cast<XPUOutTDType*>(dev_ctx->Alloc<OutType>(out)),
      in.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  dev_ctx->Wait();
}

template <typename InType>
static void XPUTransDataType(
    const phi::DenseTensor& in,
    phi::DenseTensor* out,
    const paddle::framework::proto::VarType::Type& dst_type,
    const phi::DeviceContext* ctx) {
  auto* context = static_cast<const phi::XPUContext*>(ctx);

#define XPUCastCallback(cpp_type, proto_type)          \
  do {                                                 \
    if (dst_type == proto_type) {                      \
      XPUCastData<InType, cpp_type>(in, out, context); \
    }                                                  \
  } while (0)

  if (dst_type == proto::VarType::FP32 || dst_type == proto::VarType::FP16 ||
      dst_type == proto::VarType::BOOL || dst_type == proto::VarType::INT16 ||
      dst_type == proto::VarType::INT32 || dst_type == proto::VarType::INT64 ||
      dst_type == proto::VarType::FP64) {
    _ForEachDataTypeForXPU_(XPUCastCallback);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Data type (%s) is not supported in XPU when casting data type.",
        VarDataTypeToString(dst_type)));
  }
}

#endif

template <typename InType>
struct CastDataType {
  CastDataType(const phi::DenseTensor& in,
               phi::DenseTensor* out,
               const phi::DeviceContext* ctx)
      : in_(in), out_(out), ctx_(ctx) {}
  const phi::DenseTensor in_;
  phi::DenseTensor* out_;
  const phi::DeviceContext* ctx_;

  template <typename OutType>
  void apply() {
    auto* in_begin = in_.data<InType>();
    auto* in_end = in_begin + in_.numel();
    auto* out_begin = ctx_->Alloc<OutType>(out_);

    if (phi::is_cpu_place(in_.place())) {
      phi::Transform<phi::CPUContext> trans;
      auto* context = static_cast<const phi::CPUContext*>(ctx_);
      trans(*context,
            in_begin,
            in_end,
            out_begin,
            CastDataTypeFunctor<InType, OutType>());
#if defined(__NVCC__) || defined(__HIPCC__)
    } else if (phi::is_gpu_place(in_.place())) {
      phi::Transform<phi::GPUContext> trans;
      auto* context = static_cast<const phi::GPUContext*>(ctx_);
      trans(*context,
            in_begin,
            in_end,
            out_begin,
            CastDataTypeFunctor<InType, OutType>());
      context->Wait();
#endif
#if defined(PADDLE_WITH_IPU)
    } else if (phi::is_ipu_place(in_.place())) {
      phi::Transform<phi::CPUContext> trans;
      auto* context = static_cast<const phi::CPUContext*>(ctx_);
      trans(*context,
            in_begin,
            in_end,
            out_begin,
            CastDataTypeFunctor<InType, OutType>());
#endif
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Place type is not supported when casting data type."));
    }
  }
};

void TransDataType(const phi::KernelKey& kernel_type_for_var,
                   const phi::KernelKey& expected_kernel_type,
                   const phi::DenseTensor& in,
                   phi::DenseTensor* out) {
  PADDLE_ENFORCE_EQ(in.dtype(),
                    kernel_type_for_var.dtype(),
                    common::errors::InvalidArgument(
                        "The src dtype(%s) of input tensor and kernel_type(%s) "
                        "are not consistent.",
                        DataTypeToString(in.dtype()),
                        DataTypeToString(kernel_type_for_var.dtype())));
  auto dst_type =
      phi::TransToProtoVarTypeReturnType(expected_kernel_type.dtype());
  TransDataType(in, dst_type, out);
}

void TransDataType(const phi::DenseTensor& in,
                   const paddle::framework::proto::VarType::Type& type,
                   phi::DenseTensor* out) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();

  out->Resize(in.dims());
  auto src_type = phi::TransToProtoVarTypeReturnType(in.dtype());
  auto dst_type = type;
  auto ctx = pool.Get(in.place());

#if defined(PADDLE_WITH_XPU)
  if (phi::is_xpu_place(in.place())) {
    switch (src_type) {
      case proto::VarType::FP16:
        XPUTransDataType<phi::dtype::float16>(in, out, dst_type, ctx);
        break;
      case proto::VarType::FP32:
        XPUTransDataType<float>(in, out, dst_type, ctx);
        break;
      case proto::VarType::FP64:
        XPUTransDataType<double>(in, out, dst_type, ctx);
        break;
      case proto::VarType::BOOL:
        XPUTransDataType<bool>(in, out, dst_type, ctx);
        break;
      case proto::VarType::INT16:
        XPUTransDataType<int16_t>(in, out, dst_type, ctx);
        break;
      case proto::VarType::INT32:
        XPUTransDataType<int>(in, out, dst_type, ctx);
        break;
      case proto::VarType::INT64:
        XPUTransDataType<int64_t>(in, out, dst_type, ctx);
        break;
      default:
        PADDLE_THROW(common::errors::Unimplemented(
            "Data type (%s) is not supported in XPU when casting data type.",
            VarDataTypeToString(src_type)));
    }
    return;
  }
#endif
  switch (src_type) {
    case proto::VarType::FP16:
      phi::VisitDataType(dst_type,
                         CastDataType<phi::dtype::float16>(in, out, ctx));
      break;
    case proto::VarType::BF16:
      phi::VisitDataType(dst_type,
                         CastDataType<phi::dtype::bfloat16>(in, out, ctx));
      break;
    case proto::VarType::FP8_E4M3FN:
      phi::VisitDataType(
          dst_type, CastDataType<::phi::dtype::float8_e4m3fn>(in, out, ctx));
      break;
    case proto::VarType::FP8_E5M2:
      phi::VisitDataType(dst_type,
                         CastDataType<::phi::dtype::float8_e5m2>(in, out, ctx));
      break;
    case proto::VarType::FP32:
      phi::VisitDataType(dst_type, CastDataType<float>(in, out, ctx));
      break;
    case proto::VarType::FP64:
      phi::VisitDataType(dst_type, CastDataType<double>(in, out, ctx));
      break;
    case proto::VarType::INT32:
      phi::VisitDataType(dst_type, CastDataType<int>(in, out, ctx));
      break;
    case proto::VarType::INT64:
      phi::VisitDataType(dst_type, CastDataType<int64_t>(in, out, ctx));
      break;
    case proto::VarType::BOOL:
      phi::VisitDataType(dst_type, CastDataType<bool>(in, out, ctx));
      break;
    case proto::VarType::INT16:
      phi::VisitDataType(dst_type, CastDataType<int16_t>(in, out, ctx));
      break;
    case proto::VarType::UINT8:
      phi::VisitDataType(dst_type, CastDataType<uint8_t>(in, out, ctx));
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          VarDataTypeToString(src_type)));
  }
}

}  // namespace phi
