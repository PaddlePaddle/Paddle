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

#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/operators/reduce_ops/reduce_min_max_op.h"

namespace paddle {
namespace operators {

template <typename T>
class ReduceMaxMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<phi::DenseTensor>("X");
    auto* output = context.Output<phi::DenseTensor>("Out");
    int out_dtype = context.Attr<int>("out_dtype");
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto dims = context.Attr<std::vector<int>>("dim");
    auto input_dims = input->dims();
    const auto& input_dim_size = input->dims().size();
    std::vector<int> reduce_dims;
    if (reduce_all) {
      for (int i = 0; i < input_dims.size(); i++) {
        reduce_dims.push_back(static_cast<int>(i));
      }
    } else {
      for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] < 0) {
          reduce_dims.push_back(dims[i] + input_dim_size);
        } else {
          reduce_dims.push_back(dims[i]);
        }
      }
    }

    auto place = context.GetPlace();
    phi::DenseTensor cast_out(input->type());
    cast_out.Resize(output->dims());
    cast_out.mutable_data<T>(place);

    auto cast_out_dtype = framework::TransToProtoVarType(input->dtype());

    if (out_dtype != -1) {
      cast_out_dtype = static_cast<framework::proto::VarType::Type>(out_dtype);
    }
    if (framework::TransToProtoVarType(input->type()) != cast_out_dtype) {
      if (cast_out_dtype == framework::proto::VarType::FP32) {
        output->mutable_data<float>(place);
      } else if (cast_out_dtype == framework::proto::VarType::FP16) {
        output->mutable_data<paddle::platform::float16>(place);
      } else if (cast_out_dtype == framework::proto::VarType::INT32) {
        output->mutable_data<int32_t>(place);
      }
    } else {
      output->ShareDataWith(cast_out);
    }

    MLUCnnlTensorDesc input_desc(
        *input, CNNL_LAYOUT_ARRAY, ToCnnlDataType(input->dtype()));
    MLUCnnlTensorDesc output_desc(
        *output, CNNL_LAYOUT_ARRAY, ToCnnlDataType(output->dtype()));

    MLUCnnlReduceDesc reduction_desc(reduce_dims,
                                     CNNL_REDUCE_MAX,
                                     ToCnnlDataType<T>(),
                                     CNNL_NOT_PROPAGATE_NAN,
                                     CNNL_REDUCE_NO_INDICES,
                                     CNNL_32BIT_INDICES);

    MLUCnnl::Reduce(context,
                    true /*need_workspace*/,
                    reduction_desc.get(),
                    nullptr,
                    input_desc.get(),
                    GetBasePtr(input),
                    0 /*indices_size*/,
                    nullptr,
                    nullptr,
                    output_desc.get(),
                    GetBasePtr(output));
  }
};

template <typename T>
class ReduceMaxGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Input<Tensor>("Out");
    auto* out_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    auto reduce_dims = context.Attr<std::vector<int>>("dim");
    bool reduce_all = context.Attr<bool>("reduce_all");
    int in_dtype = context.Attr<int>("in_dtype");

    PADDLE_ENFORCE_EQ(
        in_dtype == -1,
        true,
        platform::errors::InvalidArgument(
            "MLU only support in_dtype == -1 in reduce_max_grad op."));
    auto* x_grad = context.Output<Tensor>(framework::GradVarName("X"));
    x_grad->mutable_data<T>(context.GetPlace());

    auto place = context.GetPlace();

    // broadcast
    auto x_dims_vec = phi::vectorize(x->dims());
    if (reduce_all) {
      reduce_dims.clear();
      for (size_t d = 0; d < x_dims_vec.size(); ++d) {
        reduce_dims.push_back(static_cast<int>(d));
      }
    }

    Tensor tmp_out, tmp_out_grad;
    auto tmp_out_dims_vec = x_dims_vec;
    for (auto d : reduce_dims) {
      if (d < 0) {
        d += x_dims_vec.size();
      }
      tmp_out_dims_vec[d] = 1;
    }

    tmp_out.ShareDataWith(*out);
    tmp_out.Resize(phi::make_ddim(tmp_out_dims_vec));
    tmp_out_grad.ShareDataWith(*out_grad);
    tmp_out_grad.Resize(phi::make_ddim(tmp_out_dims_vec));

    Tensor transformed_out(x->type());
    transformed_out.Resize(phi::make_ddim(x_dims_vec));
    transformed_out.mutable_data<T>(place);

    MLUCnnlTensorDesc tmp_out_desc(tmp_out);
    MLUCnnlTensorDesc transformed_out_desc(transformed_out);

    MLUCnnl::BroadcastTo(context,
                         tmp_out_desc.get(),
                         GetBasePtr(&tmp_out),
                         transformed_out_desc.get(),
                         GetBasePtr(&transformed_out));

    Tensor transformed_out_grad(x->type());
    transformed_out_grad.Resize(phi::make_ddim(x_dims_vec));
    transformed_out_grad.mutable_data<T>(place);
    MLUCnnlTensorDesc tmp_out_grad_desc(tmp_out_grad);
    MLUCnnlTensorDesc transformed_out_grad_desc(transformed_out_grad);

    MLUCnnl::BroadcastTo(context,
                         tmp_out_grad_desc.get(),
                         GetBasePtr(&tmp_out_grad),
                         transformed_out_grad_desc.get(),
                         GetBasePtr(&transformed_out_grad));

    // compare
    Tensor equal_cond;
    equal_cond.mutable_data<bool>(x_grad->dims(), place);

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc equal_cond_desc(equal_cond);

    MLUCnnl::Logic(context,
                   CNNL_LOGIC_OP_EQ,
                   x_desc.get(),
                   GetBasePtr(x),
                   transformed_out_desc.get(),
                   GetBasePtr(&transformed_out),
                   equal_cond_desc.get(),
                   GetBasePtr(&equal_cond));

    // select
    Tensor t_zero;
    t_zero.mutable_data<T>(x_grad->dims(), place);
    FillMLUTensorWithHostValue<T>(context, static_cast<T>(0), &t_zero);
    t_zero.Resize(x_grad->dims());

    MLUCnnlTensorDesc t_zero_desc(t_zero);
    MLUCnnlTensorDesc x_grad_desc(*x_grad);

    MLUCnnl::Select(context,
                    equal_cond_desc.get(),
                    GetBasePtr(&equal_cond),
                    transformed_out_grad_desc.get(),
                    GetBasePtr(&transformed_out_grad),
                    t_zero_desc.get(),
                    GetBasePtr(&t_zero),
                    x_grad_desc.get(),
                    GetBasePtr(x_grad));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(reduce_max,
                       ops::ReduceMaxMLUKernel<float>,
                       ops::ReduceMaxMLUKernel<plat::float16>,
                       ops::ReduceMaxMLUKernel<int>);
REGISTER_OP_MLU_KERNEL(reduce_max_grad,
                       ops::ReduceMaxGradMLUKernel<float>,
                       ops::ReduceMaxGradMLUKernel<plat::float16>,
                       ops::ReduceMaxGradMLUKernel<int>);
