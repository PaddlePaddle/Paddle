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

#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUWhereIndexKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx =
        context.template device_context<platform::NPUDeviceContext>();
    auto* condition = context.Input<phi::DenseTensor>("Condition");
    auto* out = context.Output<phi::DenseTensor>("Out");

    auto dims = condition->dims();
    const int rank = dims.size();

    auto place = context.GetPlace();
    const aclrtStream& stream = dev_ctx.stream();

    // Run Cast and ReduceSum to get 0 dim of Out
    phi::DenseTensor booled_cond;
    if (framework::TransToProtoVarType(condition->dtype()) !=
        framework::proto::VarType::BOOL) {
      auto bool_type = ConvertToNpuDtype(framework::proto::VarType::BOOL);
      booled_cond.mutable_data<bool>(dims, place);
      const auto& booled_runner =
          NpuOpRunner("Cast",
                      {*condition},
                      {booled_cond},
                      {{"dst_type", static_cast<int>(bool_type)}});
      booled_runner.Run(stream);
    } else {
      booled_cond.ShareDataWith(*condition);
    }
    phi::DenseTensor casted_cond;
    auto dst_dtype = ConvertToNpuDtype(framework::proto::VarType::INT64);
    casted_cond.mutable_data<int64_t>(dims, place);
    const auto& cast_runner =
        NpuOpRunner("Cast",
                    {booled_cond},
                    {casted_cond},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    cast_runner.Run(stream);

    phi::DenseTensor sumed_true_num;
    sumed_true_num.mutable_data<int64_t>({1}, place);
    phi::DenseTensor cond_axes;
    cond_axes.mutable_data<int>({dims.size()}, place);
    std::vector<int> axes_vec;
    for (int i = 0; i < dims.size(); ++i) {
      axes_vec.push_back(i);
    }
    framework::TensorFromVector<int>(axes_vec, dev_ctx, &cond_axes);
    const auto& sum_runner = NpuOpRunner("ReduceSum",
                                         {casted_cond, cond_axes},
                                         {sumed_true_num},
                                         {{"keep_dims", false}});
    sum_runner.Run(stream);

    phi::DenseTensor local_true_num;
    paddle::framework::TensorCopySync(
        sumed_true_num, platform::CPUPlace(), &local_true_num);
    auto true_num = *local_true_num.data<int64_t>();

    out->Resize(phi::make_ddim({true_num, rank}));
    out->mutable_data<int64_t>(place);

    if (true_num == 0) {
      return;
    }

    out->set_layout(DataLayout::kAnyLayout);
    NpuOpRunner runner{"Where", {*condition}, {*out}};
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(where_index,
                       ops::NPUWhereIndexKernel<int64_t>,
                       ops::NPUWhereIndexKernel<int>,
                       ops::NPUWhereIndexKernel<bool>,
                       ops::NPUWhereIndexKernel<float>,
                       ops::NPUWhereIndexKernel<double>);
