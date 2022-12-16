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

namespace paddle {
namespace operators {

template <typename T>
class MaskedSelectedMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto input = ctx.Input<phi::DenseTensor>("X");
    auto mask = ctx.Input<phi::DenseTensor>("Mask");
    auto out = ctx.Output<phi::DenseTensor>("Y");

    auto input_dim = input->dims();
    auto mask_dim = mask->dims();
    PADDLE_ENFORCE_EQ(
        input_dim,
        mask_dim,
        platform::errors::InvalidArgument(
            "The dim size of input and mask in OP(masked_selected) "
            "must be equal, but got input dim:(%ld), mask dim: "
            "(%ld). Please check input "
            "value.",
            input_dim,
            mask_dim));

    phi::DenseTensor number(framework::TransToPhiDataType(VT::INT32));
    void* number_ptr = number.mutable_data<int32_t>({1}, ctx.GetPlace());

    out->Resize(mask->dims());
    out->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc mask_desc(*mask);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::Mask(ctx,
                  CNNL_MASKED_SELECT,
                  input_desc.get(),
                  GetBasePtr(input),
                  mask_desc.get(),
                  GetBasePtr(mask),
                  nullptr,
                  nullptr,
                  out_desc.get(),
                  GetBasePtr(out),
                  static_cast<uint32_t*>(number_ptr));
  }
};

template <typename T>
class MaskedSelectedGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto mask = ctx.Input<phi::DenseTensor>("Mask");
    auto y_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    auto x_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MLUDeviceContext>();
    phi::DenseTensor mask_int32, out_size;
    std::vector<int32_t> out_size_vec;
    mask_int32.mutable_data<int32_t>(mask->dims(), ctx.GetPlace());
    out_size.mutable_data<int32_t>({1}, ctx.GetPlace());

    MLUCnnlTensorDesc mask_desc(*mask);
    MLUCnnlTensorDesc mask_int32_desc(mask_int32);
    MLUCnnlTensorDesc out_size_desc(out_size);
    auto cast_type = GetCastDataType(mask->dtype(), DataType::INT32);
    MLUCnnl::Cast(ctx,
                  cast_type,
                  mask_desc.get(),
                  GetBasePtr(mask),
                  mask_int32_desc.get(),
                  GetBasePtr(&mask_int32));

    auto mask_int32_dim = phi::vectorize(mask_int32.dims());
    std::vector<int32_t> reduce_dims;
    for (size_t i = 0; i < mask_int32_dim.size(); i++) {
      reduce_dims.push_back(static_cast<int>(i));
    }

    std::string reduce_name = "reduce_sum";
    cnnlReduceOp_t reduce_op = GetMLUCnnlReduceOp(reduce_name);
    MLUCnnlReduceDesc reduce_desc(reduce_dims,
                                  reduce_op,
                                  ToCnnlDataType<int32_t>(),
                                  CNNL_NOT_PROPAGATE_NAN,
                                  CNNL_REDUCE_NO_INDICES,
                                  CNNL_32BIT_INDICES);

    MLUCnnl::Reduce(ctx,
                    true,
                    reduce_desc.get(),
                    nullptr,
                    mask_int32_desc.get(),
                    GetBasePtr(&mask_int32),
                    0,
                    nullptr,
                    nullptr,
                    out_size_desc.get(),
                    GetBasePtr(&out_size));

    paddle::framework::TensorToVector(out_size, dev_ctx, &out_size_vec);
    dev_ctx.Wait();

    phi::DenseTensor mask_int32_tmp;
    mask_int32_tmp.ShareDataWith(mask_int32);
    mask_int32_tmp.Resize({mask_int32.numel()});
    phi::DenseTensor topk_v2_out(framework::TransToPhiDataType(VT::INT32)),
        indices_int32(framework::TransToPhiDataType(VT::INT32));
    topk_v2_out.mutable_data<int32_t>({mask_int32.numel()}, ctx.GetPlace());
    indices_int32.mutable_data<int32_t>({mask_int32.numel()}, ctx.GetPlace());

    MLUCnnlTensorDesc topk_v2_out_desc(topk_v2_out);
    MLUCnnlTensorDesc indices_int32_desc(indices_int32);
    MLUCnnlTensorDesc mask_int32_tmp_desc(mask_int32_tmp);

    const int dim = 0;
    MLUCnnl::TopK(ctx,
                  mask_int32.numel(),
                  dim,
                  true,
                  false,
                  mask_int32_tmp_desc.get(),
                  GetBasePtr(&mask_int32_tmp),
                  topk_v2_out_desc.get(),
                  GetBasePtr(&topk_v2_out),
                  indices_int32_desc.get(),
                  GetBasePtr(&indices_int32));

    auto stream = ctx.template device_context<MLUDeviceContext>().stream();

    phi::DenseTensor indices_int32_out;
    indices_int32_out.mutable_data<int32_t>({out_size_vec[0]}, ctx.GetPlace());
    memory::Copy(ctx.GetPlace(),
                 GetBasePtr(&indices_int32_out),
                 ctx.GetPlace(),
                 GetBasePtr(&indices_int32),
                 out_size_vec[0] * sizeof(int32_t),
                 stream);

    phi::DenseTensor y_grad_tmp_out;
    y_grad_tmp_out.mutable_data<T>({out_size_vec[0]}, ctx.GetPlace());
    MLUCnnlTensorDesc y_grad_tmp_out_desc(y_grad_tmp_out);
    memory::Copy(ctx.GetPlace(),
                 GetBasePtr(&y_grad_tmp_out),
                 ctx.GetPlace(),
                 GetBasePtr(y_grad),
                 out_size_vec[0] * sizeof(T),
                 stream);

    phi::DenseTensor indices_int32_tmp;
    indices_int32_tmp.ShareDataWith(indices_int32_out);
    indices_int32_tmp.Resize({out_size_vec[0], 1});
    MLUCnnlTensorDesc indices_int32_tmp_desc(indices_int32_tmp);

    const cnnlScatterNdMode_t mode = CNNL_SCATTERND_UPDATE;
    x_grad->Resize({x_grad->numel()});
    x_grad->mutable_data<T>(ctx.GetPlace());
    MLUCnnlTensorDesc x_grad_desc(*x_grad);
    MLUCnnl::ScatterNd(ctx,
                       mode,
                       indices_int32_tmp_desc.get(),
                       GetBasePtr(&indices_int32_tmp),
                       y_grad_tmp_out_desc.get(),
                       GetBasePtr(&y_grad_tmp_out),
                       nullptr,
                       nullptr,
                       x_grad_desc.get(),
                       GetBasePtr(x_grad));
    x_grad->Resize(mask->dims());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(masked_select,
                       ops::MaskedSelectedMLUKernel<float>,
                       ops::MaskedSelectedMLUKernel<int>,
                       ops::MaskedSelectedMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(masked_select_grad,
                       ops::MaskedSelectedGradMLUKernel<float>,
                       ops::MaskedSelectedGradMLUKernel<int>,
                       ops::MaskedSelectedGradMLUKernel<plat::float16>);
