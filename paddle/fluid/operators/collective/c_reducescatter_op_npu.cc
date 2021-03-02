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

#include "paddle/fluid/operators/collective/c_reducescatter_op.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CReduceScatterOpAscendKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    int ring_id = ctx.Attr<int>("ring_id");
    std::string group = std::string(HCOM_GROUP_PREFIX) + std::to_string(ring_id);
    std::string tag = ctx.Attr<std::string>("tag");
    auto place = ctx.GetPlace();
    auto comm = platform::HCCLCommContext::Instance().Get();
    // int nranks = comm->nranks();

    auto out_dims = in->dims();
    // PADDLE_ENFORCE_EQ(out_dims[0] % nranks, 0,
    //                   platform::errors::InvalidArgument(
    //                       "The input tensor X's "
    //                       "dim[0] (%d) should be divisible by nranks(%d)",
    //                       out_dims[0], nranks));
    out_dims[0] = out_dims[0] / nranks;
    out->mutable_data<T>(out_dims, place);

    int64_t recv_numel = in->numel() / nranks;

    void* inputPtr = reinterpret_cast<void*>(const_cast<T*>(in->data<T>()));
    void* outputPtr = reinterpret_cast<void*>(const_cast<T*>(out->data<T>()));
    // const T* send_buff = in->data<T>();
    // T* recv_buff = out->data<T>();
    hcclDataType_t dtype = platform::ToHCCLDataType(in->type());

    aclrtStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::NPUDeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    VLOG(3) << "begin hccl reduce scatter, parameter is: "
      << "recv_numel: " << recv_numel
      << "dtype: " << dtype
      << "hccl_red_type: " << HCCL_REP_OP_SUM
      << ", group is: " << group
      << ", tag is " << tag;

    printf("inputPtr: %p\n", inputPtr);
    printf("outputPtr: %p\n", outputPtr);

    // printf("inputPtr: %p, %d\n", inputPtr, ((int*)inputPtr)[0]);
    // printf("outputPtr: %p, %d\n", outputPtr, ((int*)outputPtr)[0]);

    // PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::hcom_all_reduce(
        // tag.c_str(), sendbuff, recvbuff, (u64)numel, dtype, hccl_red_type, group.c_str(), (void*)stream));
    
    hcclResult_t ret = platform::dynload::hcom_reduce_scatter(
        tag.c_str(), inputPtr, outputPtr, (u64)recv_numel, dtype, HCCL_REP_OP_SUM, group.c_str(), (void*)stream);
    // aclrtCreateStream(&stream);
    PADDLE_ENFORCE_NPU_SUCCESS(ret);
    printf("%d\n", ret);
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

REGISTER_OP_NPU_KERNEL(c_reducescatter, ops::CReduceScatterOpAscendKernel<float>,
                        ops::CReduceScatterOpAscendKernel<double>,
                        ops::CReduceScatterOpAscendKernel<int>,
                        ops::CReduceScatterOpAscendKernel<int64_t>,
                        ops::CReduceScatterOpAscendKernel<plat::float16>);
