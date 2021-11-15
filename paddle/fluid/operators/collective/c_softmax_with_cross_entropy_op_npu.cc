/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_softmax_with_cross_entropy_op.h"
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/softmax_impl.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class CSoftmaxWithCrossEntropyOpNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* logits = ctx.Input<Tensor>("Logits");
    const Tensor* labels = ctx.Input<Tensor>("Label");
    Tensor* softmax = ctx.Output<Tensor>("Softmax");
    Tensor* loss = ctx.Output<Tensor>("Loss");

    const int rid = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int rank = ctx.Attr<int>("rank");

    const auto& place = ctx.GetPlace();
    auto comm = paddle::platform::HCCLCommContext::Instance().Get(rid, place);
    auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    // use global calculate stream
    aclrtStream stream =
        static_cast<platform::NPUDeviceContext*>(dev_ctx)->stream();

    // auto stream =
    //    context.template device_context<paddle::platform::NPUDeviceContext>()
    //        .stream();

    // allocate memory on device.
    softmax->mutable_data<T>(place);
    loss->mutable_data<T>(place);

    const auto& logits_dims = logits->dims();
    const auto& labels_dims = labels->dims();

    const int axis = logits_dims.size() - 1;
    const int N = SizeToAxis(axis, logits_dims);
    const int D = SizeFromAxis(axis, logits_dims);

    // Tensor logits_2d, softmax_2d, loss_2d;
    // logits_2d.ShareDataWith(*logits).Resize({N, D});
    // softmax_2d.ShareDataWith(*softmax).Resize({N, D});
    // loss_2d.ShareDataWith(*loss).Resize({N, 1});

    // step 1, obtain logit_max
    Tensor logits_max(logits->type());
    logits_max =
        ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>({N, 1}, dev_ctx);
    void* logits_max_buff = logits_max.mutable_data<T>(place);

    const auto& runner1 = NpuOpRunner("ReduceMaxD", {*logits}, {logits_max},
                                      {{"axes", 1}, {"keep_dims", True}});
    runner1.Run(stream);

    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllReduce(
        logits_max_buff, logits_max_buff, logits_max.numel(),
        platform::ToHCCLDataType(logits_max.type()), HCCL_REDUCE_MAX,
        comm->comm(), reinterpret_cast<void*>(stream)));

    // step 2, obtain logit - logit_max
    Tensor steady_logits(logits->type());
    const auto& runner2 =
        NpuOpRunner("Sub", {*logits, logits_max}, {steady_logits}, {});
    runner2.Run(stream);

    // step 3, obtain predict target
    Tensor predicted_logits;
    predicted_logits =
        ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>({N, 1}, dev_ctx);

    const int start_index = rank * D;
    const int end_index = start_index + D;

    Tensor start_tensor(labels->type());
    start_tensor.mutable_data<int>({1}, place);
    FillNpuTensorWithConstant<int>(&start_tensor, start_index);

    Tensor len_val_tensor(labels->type());
    start_tensor.mutable_data<int>({1}, place);
    FillNpuTensorWithConstant<int>(&len_val_tensor, D);

    Tensor cur_label(labels->type());
    const auto& runner3 =
        NpuOpRunner("Sub", {*labels, start_tensor}, {cur_label}, {});
    runner3.Run(stream);

    Tensor bad_label(labels->type());
    Tensor val_tensor(labels->type());
    val_tensor.mutable_data<int>({1}, place);
    FillNpuTensorWithConstant<int>(&val_tensor, -1);

    NpuOpRunner runner4;
    runner4.SetType("FillD")
        .AddInput(val_tensor)
        .AddOutput(bad_label)
        .AddAttrs({{"dims", labels->dims()}})
        .Run(stream);

    Tensor zero_tensor(labels->type());
    const auto& runner5 = NpuOpRunner("ZerosLike", {cur_label}, {zero_tensor});
    runner5.Run(stream);

    Tensor condition(framework::proto::VarType::BOOL);
    const auto& runner6 =
        NpuOpRunner("GreaterEqual", {cur_label, zero_tensor}, {condition}, {});
    runner6.Run(stream);

    Tensor valid_label(labels->type());
    const auto& runner7 = NpuOpRunner(
        "Select", {condition, cur_label, bad_label}, {valid_label}, {});
    runner7.Run(stream);

    Tensor len_tensor(labels->type());
    NpuOpRunner runner8;
    runner8.SetType("FillD")
        .AddInput(len_val_tensor)
        .AddOutput(len_tensor)
        .AddAttrs({{"dims", labels->dims()}})
        .Run(stream);

    const auto& runner9 =
        NpuOpRunner("Less", {valid_label, len_tensor}, {condition}, {});
    runner9.Run(stream);

    const auto& runner10 = NpuOpRunner(
        "Select", {condition, valid_label, bad_label}, {valid_label}, {});
    runner10.Run(stream);

    const auto& runner11 =
        NpuOpRunner("Sub", {valid_label, val_tensor}, {valid_label}, {});
    runner11.Run(stream);

    Tensor tmp_logits(logits->type());
    tmp_logits.Resize({N, D + 1});
    std::vector<framework::Tensor> inputs;
    std::vector<std::string> names;
    inputs.push_back(zero_tensor);
    names.push_back("x1");
    inputs.push_back(*logits);
    names.push_back("x2");
    NpuOpRunner runner12{
        "ConcatD",
        {inputs},
        {tmp_logits},
        {{"concat_dim", 1}, {"N", static_cast<int>(inputs.size())}}};
    runner12.AddInputNames(names);
    runner12.Run(stream);

    Tensor arr_tensor(labels->type());
    arr_tensor.Resize({N, 1}) const auto& runner13 =
        NpuOpRunner("Range", {0, N, 1}, {arr_tensor}, {});
    runner13.Run(stream);

    Tensor final_label(labels->type());
    final_label.Resize({N, 2});
    inputs.clear();
    names.clear();
    inputs.push_back(arr_tensor);
    names.push_back("x1");
    inputs.push_back(valid_label);
    names.push_back("x2");
    NpuOpRunner runner14{
        "ConcatD",
        {inputs},
        {final_label},
        {{"concat_dim", 1}, {"N", static_cast<int>(inputs.size())}}};
    runner14.AddInputNames(names);
    runner14.Run(stream);

    const auto& runner15 = NpuOpRunner("GatherNd", {tmp_logits, final_label},
                                       {predicted_logits}, {});
    runner15.Run(stream);

    void* predict_logits_buff = predicted_logits.mutable_data<T>(place);
    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllReduce(
        predict_logits_buff, predict_logits_buff, predicted_logits.numel(),
        platform::ToHCCLDataType(predicted_logits.type()), HCCL_REDUCE_SUM,
        comm->comm(), stream));

    // step 4, obtain exp(logit)
    Tensor exp_logits(logits->type());
    exp_logits.Resize(logits->dims()) const auto& runner16 =
        NpuOpRunner("Exp", {*logits}, {exp_logits},
                    {{"base", -1.0}, {"scale", 1.0}, {"shift", 0.0}});
    runner16.Run(stream);

    // step 5, obtain sum_exp_logits
    Tensor sum_exp_logits;
    sum_exp_logits =
        ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>({N, 1}, dev_ctx);
    void* sum_exp_logits_buff = sum_exp_logits.mutable_data<T>(place);

    const auto& runner17 =
        NpuOpRunner("ReduceSumD", {exp_logits}, {sum_exp_logits},
                    {{"axes", std::vector<int>(){1}}, {"keep_dims", True}});
    runner17.Run(stream);

    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::ncclAllReduce(
        sum_exp_logits_buff, sum_exp_logits_buff, sum_exp_logits.numel(),
        platform::ToHCCLDataType(sum_exp_logits.type()), HCCL_REDUCE_SUM,
        comm->comm(), stream));

    // step 6, to get loss and softmax
    const auto& runner18 =
        NpuOpRunner("Log", {sum_exp_logits}, {sum_exp_logits},
                    {{"base", -1.0}, {"scale", 1.0}, {"shift", 0.0}});
    runner18.Run(stream);

    const auto& runner19 =
        NpuOpRunner("Sub", {sum_exp_logits, predicted_logits}, {*loss}, {});
    runner19.Run(stream);

    const auto& runner20 = NpuOpRunner("Mul", {*x, *y}, {*out}, {});
    runner20.Run(stream);

    const auto& runner21 =
        NpuOpRunner("Div", {exp_logits, sum_exp_logits}, {*softmax}, {});
    runner21.Run(stream);
  }
};

template <typename T>
class CSoftmaxWithCrossEntropyGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* labels = context.Input<Tensor>("Label");
    const Tensor* loss_grad =
        context.Input<Tensor>(framework::GradVarName("Loss"));
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    const Tensor* softmax = context.Input<Tensor>("Softmax");
    const int rank = context.Attr<int>("rank");
    auto& dev_ctx =
        context.template device_context<platform::NPUDeviceContext>();

    if (logit_grad != softmax) {
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), logit_grad);
    }

    aclrtStream stream =
        static_cast<platform::NPUDeviceContext*>(dev_ctx)->stream();

    const auto sofrmax_dims = softmax->dims();
    const int axis = sofrmax_dims.size() - 1;
    const int N = SizeToAxis(axis, sofrmax_dims);
    const int D = SizeFromAxis(axis, sofrmax_dims);

    Tensor logit_grad_2d;
    logit_grad_2d.ShareDataWith(*logit_grad).Resize({N, D});

    const auto& label_type = labels->type();
    const int start_index = rank * D;
    const int end_index = start_index + D;

    Tensor start_tensor(labels->type());
    start_tensor.mutable_data<int>({1}, place);
    FillNpuTensorWithConstant<int>(&start_tensor, start_index);

    Tensor len_val_tensor(labels->type());
    start_tensor.mutable_data<int>({1}, place);
    FillNpuTensorWithConstant<int>(&len_val_tensor, D);

    Tensor cur_label(labels->type());
    const auto& runner1 =
        NpuOpRunner("Sub", {*labels, start_tensor}, {cur_label}, {});
    runner1.Run(stream);

    Tensor bad_label(labels->type());
    Tensor val_tensor(labels->type());
    val_tensor.mutable_data<int>({1}, place);
    FillNpuTensorWithConstant<int>(&val_tensor, -1);

    NpuOpRunner runner2;
    runner2.SetType("FillD")
        .AddInput(val_tensor)
        .AddOutput(bad_label)
        .AddAttrs({{"dims", labels->dims()}})
        .Run(stream);

    Tensor zero_tensor(labels->type());
    const auto& runner3 = NpuOpRunner("ZerosLike", {cur_label}, {zero_tensor});
    runner3.Run(stream);

    Tensor condition(framework::proto::VarType::BOOL);
    const auto& runner4 =
        NpuOpRunner("GreaterEqual", {cur_label, zero_tensor}, {condition}, {});
    runner4.Run(stream);

    Tensor valid_label(labels->type());
    const auto& runner5 = NpuOpRunner(
        "Select", {condition, cur_label, bad_label}, {valid_label}, {});
    runner5.Run(stream);

    Tensor len_tensor(labels->type());
    NpuOpRunner runner6;
    runner6.SetType("FillD")
        .AddInput(len_val_tensor)
        .AddOutput(len_tensor)
        .AddAttrs({{"dims", labels->dims()}})
        .Run(stream);

    const auto& runner7 =
        NpuOpRunner("Less", {valid_label, len_tensor}, {condition}, {});
    runner7.Run(stream);

    const auto& runner8 = NpuOpRunner(
        "Select", {condition, valid_label, bad_label}, {valid_label}, {});
    runner8.Run(stream);

    const auto& runner9 =
        NpuOpRunner("Sub", {valid_label, val_tensor}, {valid_label}, {});
    runner9.Run(stream);

    Tensor tmp_logits(logit_grad->type());
    tmp_logits.Resize({N, D + 1});
    std::vector<framework::Tensor> inputs;
    std::vector<std::string> names;
    inputs.push_back(zero_tensor);
    names.push_back("x1");
    inputs.push_back(*logit_grad);
    names.push_back("x2");
    NpuOpRunner runner10{
        "ConcatD",
        {inputs},
        {tmp_logits},
        {{"concat_dim", 1}, {"N", static_cast<int>(inputs.size())}}};
    runner10.AddInputNames(names);
    runner10.Run(stream);

    Tensor arr_tensor(labels->type());
    arr_tensor.Resize({N, 1}) const auto& runner11 =
        NpuOpRunner("Range", {0, N, 1}, {arr_tensor}, {});
    runner11.Run(stream);

    Tensor final_label(labels->type());
    final_label.Resize({N, 2});
    inputs.clear();
    names.clear();
    inputs.push_back(arr_tensor);
    names.push_back("x1");
    inputs.push_back(valid_label);
    names.push_back("x2");
    NpuOpRunner runner12{
        "ConcatD",
        {inputs},
        {final_label},
        {{"concat_dim", 1}, {"N", static_cast<int>(inputs.size())}}};
    runner12.AddInputNames(names);
    runner12.Run(stream);

    // to build two matrix for getting final grad.
    Tensor eq_mat(logit_grad->type());
    eq_mat.Resize(tmp_logits.dims());

    Tensor tmp_mat(logit_grad->type());
    tmp_mat.Resize(tmp_logits.dims());
    const auto& runner13 =
        NpuOpRunner("Add", {tmp_logits, val_tensor}, {tmp_mat}, {});
    runner13.Run(stream);

    const auto& runner14 =
        NpuOpRunner("Mul", {tmp_mat, loss_grad}, {eq_mat}, {});
    runner14.Run(stream);

    Tensor ne_mat(logit_grad->type());
    ne_mat.Resize(tmp_logits.dims());
    const auto& runner15 =
        NpuOpRunner("Mul", {tmp_logits, loss_grad}, {ne_mat}, {});
    runner15.Run(stream);

    Tensor zero_mat(tmp_logits->type());
    const auto& runner16 = NpuOpRunner("ZerosLike", {tmp_logits}, {zero_mat});
    runner16.Run(stream);

    Tensor eq_logit(logit_grad->type());
    cur_rank_logit.Resize({N, 1});
    const auto& runner17 =
        NpuOpRunner("GatherNd", {eq_mat, final_label}, {eq_logit}, {});
    runner17.Run(stream);

    Tensor ne_logit(logit_grad->type());
    ne_logit.Resize({N, 1});
    const auto& runner18 =
        NpuOpRunner("GatherNd", {ne_mat, final_label}, {ne_logit}, {});
    runner18.Run(stream);

    Tensor eq_ne_logit(tmp_logits.type());
    eq_ne_logit.Resize({N, 1});
    const auto& runner19 =
        NpuOpRunner("Sub", {eq_logit, ne_logit}, {eq_ne_logit}, {});
    runner19.Run(stream);

    const auto& runner20 = NpuOpRunner(
        "ScatterNd", {final_label, eq_ne_logit, std::vector<int>(){N, D + 1}},
        {zero_mat}, {});
    runner20.Run(stream);

    const auto& runner21 =
        NpuOpRunner("Add", {zero_mat, ne_mat}, {*logit_grad}, {});
    runner21.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle
