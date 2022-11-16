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

#include "paddle/fluid/framework/data_type_transform.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/platform/transform.h"

#if defined(PADDLE_WITH_XPU)
#include "paddle/fluid/platform/device/device_wrapper.h"
#endif

namespace paddle {
namespace framework {

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
                        const platform::XPUDeviceContext* dev_ctx) {
  using XPUInTDType = typename XPUTypeTrait<InType>::Type;
  using XPUOutTDType = typename XPUTypeTrait<OutType>::Type;
  int r = xpu::cast_v2<XPUInTDType, XPUOutTDType>(
      dev_ctx->x_context(),
      reinterpret_cast<const XPUInTDType*>(in.data<InType>()),
      reinterpret_cast<XPUOutTDType*>(out->mutable_data<OutType>(in.place())),
      in.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast_v2");
  dev_ctx->Wait();
}

template <typename InType>
static void XPUTransDataType(
    const phi::DenseTensor& in,
    phi::DenseTensor* out,
    const paddle::framework::proto::VarType::Type& dst_type,
    const platform::DeviceContext* ctx) {
  auto* context = static_cast<const platform::XPUDeviceContext*>(ctx);

#define XPUCastCallback(cpp_type, proto_type)          \
  do {                                                 \
    if (dst_type == proto_type) {                      \
      XPUCastData<InType, cpp_type>(in, out, context); \
    }                                                  \
  } while (0)

  if (dst_type == proto::VarType::FP32 && dst_type == proto::VarType::FP16 &&
      dst_type == proto::VarType::BOOL && dst_type == proto::VarType::INT16 &&
      dst_type == proto::VarType::INT32 && dst_type == proto::VarType::INT64) {
    _ForEachDataType_(XPUCastCallback);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Data type (%s) is not supported in XPU when casting data type.",
        DataTypeToString(dst_type)));
  }
}

#endif

template <typename InType>
struct CastDataType {
  CastDataType(const phi::DenseTensor& in,
               phi::DenseTensor* out,
               const platform::DeviceContext* ctx)
      : in_(in), out_(out), ctx_(ctx) {}
  const phi::DenseTensor in_;
  phi::DenseTensor* out_;
  const platform::DeviceContext* ctx_;

  template <typename OutType>
  void apply() {
    auto* in_begin = in_.data<InType>();
    auto* in_end = in_begin + in_.numel();
    auto* out_begin = out_->mutable_data<OutType>(in_.place());

    if (platform::is_cpu_place(in_.place())) {
      platform::Transform<phi::CPUContext> trans;
      auto* context = static_cast<const phi::CPUContext*>(ctx_);
      trans(*context,
            in_begin,
            in_end,
            out_begin,
            CastDataTypeFunctor<InType, OutType>());
#if defined(__NVCC__) || defined(__HIPCC__)
    } else if (platform::is_gpu_place(in_.place())) {
      platform::Transform<phi::GPUContext> trans;
      auto* context = static_cast<const phi::GPUContext*>(ctx_);
      trans(*context,
            in_begin,
            in_end,
            out_begin,
            CastDataTypeFunctor<InType, OutType>());
      context->Wait();
#endif
#if defined(PADDLE_WITH_IPU)
    } else if (platform::is_ipu_place(in_.place())) {
      platform::Transform<phi::CPUContext> trans;
      auto* context = static_cast<const phi::CPUContext*>(ctx_);
      trans(*context,
            in_begin,
            in_end,
            out_begin,
            CastDataTypeFunctor<InType, OutType>());
#endif
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Place type is not supported when casting data type."));
    }
  }
};

void TransDataType(const OpKernelType& kernel_type_for_var,
                   const OpKernelType& expected_kernel_type,
                   const phi::DenseTensor& in,
                   phi::DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      framework::TransToProtoVarType(in.dtype()),
      kernel_type_for_var.data_type_,
      platform::errors::InvalidArgument(
          "The src dtype(%s) of input tensor and kernel_type(%s) "
          "are not conststent.",
          DataTypeToString(framework::TransToProtoVarType(in.dtype())),
          DataTypeToString(kernel_type_for_var.data_type_)));
  auto dst_type = expected_kernel_type.data_type_;
  TransDataType(in, dst_type, out);
}

void TransDataType(const phi::DenseTensor& in,
                   const paddle::framework::proto::VarType::Type& type,
                   phi::DenseTensor* out) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

  out->Resize(in.dims());
  auto src_type = framework::TransToProtoVarType(in.dtype());
  auto dst_type = type;
  auto ctx = pool.Get(in.place());

#if defined(PADDLE_WITH_XPU)
  switch (src_type) {
    case proto::VarType::FP16:
      XPUTransDataType<platform::float16>(in, out, dst_type, ctx);
      break;
    case proto::VarType::FP32:
      XPUTransDataType<float>(in, out, dst_type, ctx);
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
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported in XPU when casting data type.",
          DataTypeToString(src_type)));
  }

#else

  switch (src_type) {
    case proto::VarType::FP16:
      framework::VisitDataType(dst_type,
                               CastDataType<platform::float16>(in, out, ctx));
      break;
    case proto::VarType::BF16:
      framework::VisitDataType(dst_type,
                               CastDataType<platform::bfloat16>(in, out, ctx));
      break;
    case proto::VarType::FP32:
      framework::VisitDataType(dst_type, CastDataType<float>(in, out, ctx));
      break;
    case proto::VarType::FP64:
      framework::VisitDataType(dst_type, CastDataType<double>(in, out, ctx));
      break;
    case proto::VarType::INT32:
      framework::VisitDataType(dst_type, CastDataType<int>(in, out, ctx));
      break;
    case proto::VarType::INT64:
      framework::VisitDataType(dst_type, CastDataType<int64_t>(in, out, ctx));
      break;
    case proto::VarType::BOOL:
      framework::VisitDataType(dst_type, CastDataType<bool>(in, out, ctx));
      break;
    case proto::VarType::INT16:
      framework::VisitDataType(dst_type, CastDataType<int16_t>(in, out, ctx));
      break;
    case proto::VarType::UINT8:
      framework::VisitDataType(dst_type, CastDataType<uint8_t>(in, out, ctx));
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          DataTypeToString(src_type)));
  }
#endif
}

void TransComplexToReal(const proto::VarType::Type& dst_type,
                        const proto::VarType::Type& src_type,
                        const phi::DenseTensor& in,
                        phi::DenseTensor* out) {
  auto& pool = platform::DeviceContextPool::Instance();
  auto* ctx = pool.Get(in.place());
  out->Resize(in.dims());
  // complex -> real
  switch (src_type) {
    case proto::VarType::COMPLEX64:
      framework::VisitDataType(
          dst_type, CastDataType<platform::complex<float>>(in, out, ctx));
      break;
    case proto::VarType::COMPLEX128:
      framework::VisitDataType(
          dst_type, CastDataType<platform::complex<double>>(in, out, ctx));
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when casting complex tensor to real "
          "data type.",
          DataTypeToString(src_type)));
  }
}

}  // namespace framework
}  // namespace paddle
