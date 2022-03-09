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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class AccuracyMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* indices = ctx.Input<Tensor>("Indices");
    auto* label = ctx.Input<Tensor>("Label");

    auto* accuracy = ctx.Output<Tensor>("Accuracy");
    auto* correct = ctx.Output<Tensor>("Correct");
    auto* total = ctx.Output<Tensor>("Total");

    int num_samples = indices->dims()[0];
    if (num_samples == 0) {
      return;
    }

    // cast `indices` or `label` if their type is not INT32
    Tensor indices_int32(framework::TransToPhiDataType(VT::INT32));
    Tensor label_int32(framework::TransToPhiDataType(VT::INT32));
    auto indices_type = framework::TransToProtoVarType(indices->type());
    if (indices_type != VT::INT32) {
      PADDLE_ENFORCE_EQ(MLUSupportsCast(indices_type, VT::INT32), true,
                        platform::errors::Unimplemented(
                            "In accuracy mlu kernel, cast indices from [%s] to "
                            "[%s] is not supported.",
                            framework::DataTypeToString(indices_type),
                            framework::DataTypeToString(VT::INT32)));
      indices_int32.Resize(indices->dims());
      indices_int32.mutable_data<int>(ctx.GetPlace());
      MLUCnnlTensorDesc org_indices_desc(*indices);
      MLUCnnlTensorDesc indices_int32_desc(indices_int32);
      cnnlCastDataType_t cast_type = GetCastDataType(indices_type, VT::INT32);
      MLUCnnl::Cast(ctx, cast_type, org_indices_desc.get(), GetBasePtr(indices),
                    indices_int32_desc.get(), GetBasePtr(&indices_int32));
    } else {
      indices_int32.ShareDataWith(*indices);
    }
    auto label_type = framework::TransToProtoVarType(label->type());
    if (label_type != VT::INT32) {
      PADDLE_ENFORCE_EQ(
          MLUSupportsCast(label_type, VT::INT32), true,
          platform::errors::Unimplemented(
              "In accuracy mlu kernel, cast label from [%s] to [%s] "
              "is not supported.",
              framework::DataTypeToString(label_type),
              framework::DataTypeToString(VT::INT32)));
      label_int32.Resize(label->dims());
      label_int32.mutable_data<int>(ctx.GetPlace());
      MLUCnnlTensorDesc org_label_desc(*label);
      MLUCnnlTensorDesc label_int32_desc(label_int32);
      cnnlCastDataType_t cast_type = GetCastDataType(label_type, VT::INT32);
      MLUCnnl::Cast(ctx, cast_type, org_label_desc.get(), GetBasePtr(label),
                    label_int32_desc.get(), GetBasePtr(&label_int32));
    } else {
      label_int32.ShareDataWith(*label);
    }

    // equal
    MLUCnnlTensorDesc indices_int32_desc(indices_int32);
    MLUCnnlTensorDesc label_int32_desc(label_int32);
    Tensor equal_tensor(framework::TransToPhiDataType(VT::BOOL));
    equal_tensor.Resize(indices->dims());
    equal_tensor.mutable_data<bool>(ctx.GetPlace());
    MLUCnnlTensorDesc equal_tensor_desc(equal_tensor);
    MLUCnnl::Logic(ctx, CNNL_LOGIC_OP_EQ, indices_int32_desc.get(),
                   GetBasePtr(&indices_int32), label_int32_desc.get(),
                   GetBasePtr(&label_int32), equal_tensor_desc.get(),
                   GetBasePtr(&equal_tensor));

    // cast equal
    Tensor equal_fp32(framework::TransToPhiDataType(VT::FP32));
    equal_fp32.Resize(indices->dims());
    equal_fp32.mutable_data<float>(ctx.GetPlace());
    MLUCnnlTensorDesc equal_fp32_desc(equal_fp32);
    cnnlCastDataType_t equal_cast_type = GetCastDataType(VT::BOOL, VT::FP32);
    MLUCnnl::Cast(ctx, equal_cast_type, equal_tensor_desc.get(),
                  GetBasePtr(&equal_tensor), equal_fp32_desc.get(),
                  GetBasePtr(&equal_fp32));

    // [correct]
    // reduce_max
    Tensor correct_max(framework::TransToPhiDataType(VT::FP32));
    correct_max.Resize(phi::make_ddim({num_samples}));
    correct_max.mutable_data<float>(ctx.GetPlace());
    MLUCnnlTensorDesc correct_max_desc(correct_max);
    MLUCnnlReduceDesc reduce_max_desc(
        {1}, CNNL_REDUCE_MAX, ToCnnlDataType<float>(), CNNL_NOT_PROPAGATE_NAN,
        CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);
    MLUCnnl::Reduce(ctx, true /*need_workspace*/, reduce_max_desc.get(),
                    nullptr, equal_fp32_desc.get(), GetBasePtr(&equal_fp32),
                    0 /*indices_size*/, nullptr, nullptr,
                    correct_max_desc.get(), GetBasePtr(&correct_max));

    // reduce_sum
    Tensor correct_sum(framework::TransToPhiDataType(VT::FP32));
    correct_sum.Resize(correct->dims());
    correct_sum.mutable_data<float>(ctx.GetPlace());
    MLUCnnlTensorDesc correct_sum_desc(correct_sum);
    MLUCnnlReduceDesc reduce_sum_desc(
        {0}, CNNL_REDUCE_ADD, ToCnnlDataType<float>(), CNNL_NOT_PROPAGATE_NAN,
        CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);
    MLUCnnl::Reduce(ctx, true /*need_workspace*/, reduce_sum_desc.get(),
                    nullptr, correct_max_desc.get(), GetBasePtr(&correct_max),
                    0 /*indices_size*/, nullptr, nullptr,
                    correct_sum_desc.get(), GetBasePtr(&correct_sum));

    // cast to int
    correct->mutable_data<int>(ctx.GetPlace());
    MLUCnnlTensorDesc correct_desc(*correct);
    cnnlCastDataType_t correct_cast_type = GetCastDataType(VT::FP32, VT::INT32);
    MLUCnnl::Cast(ctx, correct_cast_type, correct_sum_desc.get(),
                  GetBasePtr(&correct_sum), correct_desc.get(),
                  GetBasePtr(correct));

    // [total]
    total->mutable_data<int>(ctx.GetPlace());
    MLUCnnlTensorDesc total_desc(*total);
    MLUCnnl::Fill(ctx, num_samples, total_desc.get(), GetBasePtr(total));

    // use `total` of type `float32` for calculating accuracy
    Tensor total_fp32(framework::TransToPhiDataType(VT::FP32));
    total_fp32.Resize(total->dims());
    total_fp32.mutable_data<float>(ctx.GetPlace());
    MLUCnnlTensorDesc total_fp32_desc(total_fp32);
    MLUCnnl::Fill(ctx, static_cast<float>(num_samples), total_fp32_desc.get(),
                  GetBasePtr(&total_fp32));

    // [accuracy]
    accuracy->mutable_data<float>(ctx.GetPlace());
    MLUCnnlTensorDesc accuracy_desc(*accuracy);
    MLUCnnl::Div(ctx, CNNL_COMPUTATION_HIGH_PRECISION, correct_sum_desc.get(),
                 GetBasePtr(&correct_sum), total_fp32_desc.get(),
                 GetBasePtr(&total_fp32), accuracy_desc.get(),
                 GetBasePtr(accuracy));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(accuracy, ops::AccuracyMLUKernel<float>,
                       ops::AccuracyMLUKernel<paddle::platform::float16>,
                       ops::AccuracyMLUKernel<int16_t>,
                       ops::AccuracyMLUKernel<int64_t>,
                       ops::AccuracyMLUKernel<uint8_t>,
                       ops::AccuracyMLUKernel<int>);
