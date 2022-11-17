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

#include "paddle/fluid/operators/collective/c_allgather_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

#if defined(PADDLE_WITH_CNCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/mlu/cncl_helper.h"
#endif
#include "paddle/fluid/framework/convert_utils.h"

namespace paddle {
namespace operators {

template <typename T>
class CAllGatherOpMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();
    auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
#if defined(PADDLE_WITH_CNCL)
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    int nranks = ctx.Attr<int>("nranks");
    int rid = ctx.Attr<int>("ring_id");
    auto comm = platform::CNCLCommContext::Instance().Get(rid, place);
    PADDLE_ENFORCE_EQ(
        nranks,
        comm->nranks(),
        platform::errors::InvalidArgument(
            "nranks: %s should equal to %s", nranks, comm->nranks()));

    framework::DDim out_dims = x->dims();
    out_dims[0] *= nranks;
    out->mutable_data<T>(out_dims, place);

    uint32_t send_numel = x->numel();
    void* send_buff;
    void* recv_buff;
    phi::DenseTensor in_tensor, out_tensor;
    if (framework::TransToProtoVarType(x->dtype()) ==
        framework::proto::VarType::INT64) {
      // cast from int64 to int32 since cncl do not support int64
      in_tensor.mutable_data<int32_t>(x->dims(), place);
      out_tensor.mutable_data<int32_t>(out->dims(), place);
      MLUCnnlTensorDesc x_int64_desc(*x);
      MLUCnnlTensorDesc x_int32_desc(in_tensor);
      cnnlCastDataType_t cast_type = GetCastDataType(VT::INT64, VT::INT32);
      MLUCnnl::Cast(ctx,
                    cast_type,
                    x_int64_desc.get(),
                    GetBasePtr(x),
                    x_int32_desc.get(),
                    GetBasePtr(&in_tensor));
      send_buff = reinterpret_cast<void*>(in_tensor.data<int32_t>());
      recv_buff = reinterpret_cast<void*>(out_tensor.data<int32_t>());
    } else {
      in_tensor.ShareDataWith(*x);
      out_tensor.ShareDataWith(*out);
      send_buff = reinterpret_cast<void*>(in_tensor.data<T>());
      recv_buff = reinterpret_cast<void*>(out_tensor.data<T>());
    }

    mluStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      stream = static_cast<platform::MLUDeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    cnclDataType_t dtype = platform::ToCNCLDataType(
        framework::TransToProtoVarType(in_tensor.dtype()));

    PADDLE_ENFORCE_MLU_SUCCESS(cnclAllGather(
        send_buff, recv_buff, send_numel, dtype, comm->comm(), stream));
    if (framework::TransToProtoVarType(x->dtype()) ==
        framework::proto::VarType::INT64) {
      // cast back from int64 out_tensor to out
      MLUCnnlTensorDesc out_int64_desc(*out);
      MLUCnnlTensorDesc out_int32_desc(out_tensor);
      cnnlCastDataType_t cast_type = GetCastDataType(VT::INT32, VT::INT64);
      MLUCnnl::Cast(ctx,
                    cast_type,
                    out_int32_desc.get(),
                    GetBasePtr(&out_tensor),
                    out_int64_desc.get(),
                    GetBasePtr(out));
    }
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with MLU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(c_allgather,
                       ops::CAllGatherOpMLUKernel<float>,
                       ops::CAllGatherOpMLUKernel<uint8_t>,
                       ops::CAllGatherOpMLUKernel<int>,
                       ops::CAllGatherOpMLUKernel<int8_t>,
                       ops::CAllGatherOpMLUKernel<int16_t>,
                       ops::CAllGatherOpMLUKernel<int64_t>,
                       ops::CAllGatherOpMLUKernel<plat::float16>);
