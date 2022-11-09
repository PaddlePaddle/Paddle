// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class ArgMaxMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto axis = static_cast<int>(ctx.Attr<int64_t>("axis"));
    auto dtype = ctx.Attr<int>("dtype");
    const bool& flatten = ctx.Attr<bool>("flatten");

    if (x->numel() == 0) return;
    PADDLE_ENFORCE_EQ(
        (dtype == 2 || dtype == 3),
        true,
        platform::errors::InvalidArgument(
            "The attribute of dtype in argmax op must be [%s] or [%s], "
            "but "
            "received [%s]",
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT64),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT32),
            paddle::framework::DataTypeToString(
                static_cast<framework::proto::VarType::Type>(dtype))));

    if (axis < 0) {
      framework::DDim x_dims;
      x_dims = x->dims();
      axis += x_dims.size();
    }

    phi::DenseTensor flatten_x(x->type());
    flatten_x.ShareDataWith(*x);
    if (flatten) {
      flatten_x.Resize(phi::make_ddim({x->numel()}));
      // if flatten, the axis just as 0
      axis = 0;
    }
    std::vector<int> reduce_dims;
    reduce_dims.push_back(axis);

    auto out_dims = out->dims();
    int out_count = out_dims[0];
    for (int i = 1; i < out_dims.size(); i++) {
      out_count = out_count * out_dims[i];
    }
    size_t indices_size_inbytes = out_count * sizeof(int32_t);
    auto& dev_ctx = ctx.template device_context<MLUDeviceContext>();
    phi::DenseTensor value_out =
        ctx.AllocateTmpTensor<T, MLUDeviceContext>(out->dims(), dev_ctx);
    MLUCnnlTensorDesc value_out_desc(value_out);
    MLUCnnlTensorDesc input_desc(
        flatten_x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(flatten_x.dtype()));
    MLUCnnlReduceDesc reduction_desc(reduce_dims,
                                     CNNL_REDUCE_MAX,
                                     ToCnnlDataType<T>(),
                                     CNNL_NOT_PROPAGATE_NAN,
                                     CNNL_REDUCE_ONLY_INDICES,
                                     CNNL_32BIT_INDICES);

    if (dtype == 2) {
      out->template mutable_data<int32_t>(ctx.GetPlace());
      MLUCnnl::Reduce(ctx,
                      true /*need_workspace*/,
                      reduction_desc.get(),
                      nullptr,
                      input_desc.get(),
                      GetBasePtr(&flatten_x),
                      indices_size_inbytes /*indices_size*/,
                      GetBasePtr(out),
                      nullptr,
                      value_out_desc.get(),
                      GetBasePtr(&value_out));
    } else {
      out->template mutable_data<int64_t>(ctx.GetPlace());
      phi::DenseTensor out_int32 =
          ctx.AllocateTmpTensor<int32_t, MLUDeviceContext>(out->dims(),
                                                           dev_ctx);
      MLUCnnl::Reduce(ctx,
                      true /*need_workspace*/,
                      reduction_desc.get(),
                      nullptr,
                      input_desc.get(),
                      GetBasePtr(&flatten_x),
                      indices_size_inbytes /*indices_size*/,
                      GetBasePtr(&out_int32),
                      nullptr,
                      value_out_desc.get(),
                      GetBasePtr(&value_out));

      // cast indices type to int64
      MLUCnnlTensorDesc out_int32_desc(out_int32);
      MLUCnnlTensorDesc cast_output_desc(*out);
      cnnlCastDataType_t cast_type = GetCastDataType(VT::INT32, VT::INT64);
      MLUCnnl::Cast(ctx,
                    cast_type,
                    out_int32_desc.get(),
                    GetBasePtr(&out_int32),
                    cast_output_desc.get(),
                    GetBasePtr(out));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_MLU_KERNEL(arg_max,
                       ops::ArgMaxMLUKernel<int>,
                       ops::ArgMaxMLUKernel<float>,
                       ops::ArgMaxMLUKernel<paddle::platform::float16>);
