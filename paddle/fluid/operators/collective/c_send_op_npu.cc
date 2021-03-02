/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/send_v2_op.h"
#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CSendOpASCENDKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    auto x = ctx.Input<framework::LoDTensor>("X");
    auto out = ctx.Output<framework::LoDTensor>("Out");
    int numel = x->numel();
    HcclDataType dtype = platform::ToHCCLDataType(x->type());
    int ring_id = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = platform::HCCLCommContext::Instance().Get(rid, place);

    int destRank = ctx.Attr<int>("destRank")
    std::string tag = ctx.Attr<std::string>("tag");
    std::string group = std::string(HCOM_GROUP_PREFIX) + std::to_string(ring_id);
    int srTag = ctx.Attr<int>("srTag");

    VLOG(3) << "Begin hccl send, parameter is: "<< "destRank" << destRank
      << ", group is " << group
      << ", tag is " << tag
      << ", srTag is " << srTag;

    aclrtStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::NPUDeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    platform::dynload::hcom_send(
        tag.c_string(), reinterpret_cast<void*>(const_cast<T*>(x->data<T>())), numel, dtype,
        destRank, srTag, group.c_string(), stream);

    VLOG(3) << "destRank " << destRank << " invoke Send. sent "
            << x->numel();

    if (out != x) {
        framework::TensorCopy(
            *static_cast<const framework::Tensor*>(x), place,
            *platform::DeviceContextPool::Instance().Get(place),
            static_cast<framework::Tensor*>(out));
        }

    out->Resize(x->dims());
    out->set_lod(x->lod());
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(c_send, ops::CSendOpASCENDKernel<float>,
                        ops::CSendOpASCENDKernel<int>,
                        ops::CSendOpASCENDKernel<int8_t>,
                        ops::CSendOpASCENDKernel<plat::float16>);
