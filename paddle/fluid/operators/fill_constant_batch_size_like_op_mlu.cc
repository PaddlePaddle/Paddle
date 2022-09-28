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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {
template <typename T>
class FillConstantBatchSizeLikeOpMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto float_value = ctx.Attr<float>("value");
    auto str_value = ctx.Attr<std::string>("str_value");
    auto force_cpu = ctx.Attr<bool>("force_cpu");

    auto *out = ctx.Output<phi::DenseTensor>("Out");
    auto *in = ctx.Input<framework::LoDTensor>("Input");
    if (in->lod().size() && ctx.Attr<int>("input_dim_idx") == 0) {
      // set the correct batch size for the LoDTensor.
      auto odims = out->dims();
      int output_dim_idx = ctx.Attr<int>("output_dim_idx");
      odims[output_dim_idx] = static_cast<int>(in->lod().back().size()) - 1;
      out->mutable_data<T>(odims, ctx.GetPlace());
    }

    T value;
    if (str_value.empty()) {
      value = static_cast<T>(float_value);
    } else {
      // handle NaN/Inf first, which cannot be read from stream.
      if (str_value == "inf") {
        value = static_cast<T>(std::numeric_limits<double>::infinity());
      } else if (str_value == "-inf") {
        value = static_cast<T>(-std::numeric_limits<double>::infinity());
      } else if (str_value == "nan") {
        value = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
      } else {
        std::stringstream convert_stream(str_value);
        if (std::is_same<int64_t, T>::value) {
          int64_t tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        } else {
          double tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        }
      }
    }

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    bool cpu_place = force_cpu || ctx.GetPlace() == platform::CPUPlace();
    if (cpu_place) {
      auto &dev_ctx = *pool.Get(platform::CPUPlace());
      phi::funcs::SetConstant<phi::CPUContext, T> functor;
      out->mutable_data(platform::CPUPlace(),
                        framework::TransToPhiDataType(data_type));
      functor(reinterpret_cast<const phi::CPUContext &>(dev_ctx),
              out,
              static_cast<T>(value));
    } else {
      out->mutable_data(ctx.GetPlace(),
                        framework::TransToPhiDataType(data_type));
      const T *value_data = &value;
      cnnlPointerMode_t pointer_mode = CNNL_POINTER_MODE_HOST;
      MLUCnnlTensorDesc output_desc(*out);
      MLUCnnl::Fill(
          ctx, pointer_mode, value_data, output_desc.get(), GetBasePtr(out));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(
    fill_constant_batch_size_like,
    ops::FillConstantBatchSizeLikeOpMLUKernel<int>,
    ops::FillConstantBatchSizeLikeOpMLUKernel<float>,
    ops::FillConstantBatchSizeLikeOpMLUKernel<plat::float16>);
