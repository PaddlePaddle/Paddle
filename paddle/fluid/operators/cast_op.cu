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

#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/pten/include/manipulation.h"

namespace paddle {
namespace operators {

template <typename InT>
class CastCUDAOpKernel : public framework::OpKernel<InT> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");

    auto out_dtype = context.Attr<int>("out_dtype");
    auto in_dtype = context.Attr<int>("in_dtype");

    auto& dev_ctx = context.cuda_device_context();
    out->mutable_data(dev_ctx.GetPlace(),
                      static_cast<framework::proto::VarType::Type>(out_dtype));

    auto pt_x = paddle::experimental::MakePtenDenseTensor(*in);
    auto pt_out = paddle::experimental::MakePtenDenseTensor(*out);

    auto pt_out_dtype = pten::TransToPtenDataType(
        static_cast<framework::proto::VarType::Type>(out_dtype));
    auto pt_in_dtype = pten::TransToPtenDataType(
        static_cast<framework::proto::VarType::Type>(in_dtype));

    // casll new kernel
    pten::Cast<InT>(dev_ctx, *pt_x.get(), pt_out_dtype, pt_in_dtype,
                    pt_out.get());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_CAST_CUDA_BASE(op_name, ...)                               \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      op_name, ops::CastCUDAOpKernel<float>, ops::CastCUDAOpKernel<double>, \
      ops::CastCUDAOpKernel<int>, ops::CastCUDAOpKernel<int64_t>,           \
      ops::CastCUDAOpKernel<int16_t>, ops::CastCUDAOpKernel<bool>,          \
      ops::CastCUDAOpKernel<uint8_t>, ops::CastCUDAOpKernel<plat::float16>, \
      ops::CastCUDAOpKernel<plat::complex<float>>,                          \
      ops::CastCUDAOpKernel<plat::complex<double>>, ##__VA_ARGS__);

#if !defined(PADDLE_WITH_HIP)
REGISTER_CAST_CUDA_BASE(cast, ops::CastCUDAOpKernel<plat::bfloat16>)
// See [ why register transfer_dtype_op alias with cast_op? ] in cast_op.cc
REGISTER_CAST_CUDA_BASE(transfer_dtype, ops::CastCUDAOpKernel<plat::bfloat16>)
#else
REGISTER_CAST_CUDA_BASE(cast)
REGISTER_CAST_CUDA_BASE(transfer_dtype)
#endif
