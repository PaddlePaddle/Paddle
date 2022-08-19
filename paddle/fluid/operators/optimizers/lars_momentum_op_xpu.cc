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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/optimizers/lars_momentum_op.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"

namespace paddle {
namespace operators {

template <typename T>
class LarsMomentumOpXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool multi_precision = ctx.Attr<bool>("multi_precision");
    auto param_out = ctx.MultiOutput<framework::LoDTensor>("ParamOut");
    auto velocity_out = ctx.MultiOutput<framework::LoDTensor>("VelocityOut");
    auto param = ctx.MultiInput<framework::LoDTensor>("Param");
    auto velocity = ctx.MultiInput<framework::LoDTensor>("Velocity");
    auto learning_rate = ctx.MultiInput<framework::LoDTensor>("LearningRate");
    auto grad = ctx.MultiInput<framework::LoDTensor>("Grad");
    auto weight_decay_arr = ctx.Attr<std::vector<float>>("lars_weight_decay");
    auto master_param = ctx.MultiInput<framework::LoDTensor>("MasterParam");
    auto master_param_out =
        ctx.MultiOutput<framework::LoDTensor>("MasterParamOut");
    float mu = static_cast<T>(ctx.Attr<float>("mu"));
    float lars_coeff = ctx.Attr<float>("lars_coeff");
    float epsilon = ctx.Attr<float>("epsilon");
    float rescale_grad = ctx.Attr<float>("rescale_grad");

    std::vector<XPUType*> param_list;
    std::vector<XPUType*> grad_list;
    std::vector<XPUType*> param_out_list;
    std::vector<float*> velocity_list;
    std::vector<float*> velocity_out_list;
    std::vector<float*> lrs;
    std::vector<int> param_sizes;

    std::vector<float*> master_param_list;
    std::vector<float*> master_param_out_list;
    int op_num = param.size();
    for (int i = 0; i < op_num; ++i) {
      param_list.push_back(
          reinterpret_cast<XPUType*>(const_cast<T*>((param[i]->data<T>()))));
      grad_list.push_back(
          reinterpret_cast<XPUType*>(const_cast<T*>(grad[i]->data<T>())));
      param_out_list.push_back(reinterpret_cast<XPUType*>(
          param_out[i]->mutable_data<T>(ctx.GetPlace())));
      velocity_list.push_back(const_cast<float*>(velocity[i]->data<float>()));
      velocity_out_list.push_back(
          velocity_out[i]->mutable_data<float>(ctx.GetPlace()));
      lrs.push_back(const_cast<float*>(learning_rate[i]->data<float>()));
      param_sizes.push_back(param[i]->numel());

      PADDLE_ENFORCE_EQ(
          param_list[i],
          param_out_list[i],
          platform::errors::InvalidArgument(
              "Input(Param) and Output(ParamOut) must be the same Tensors."));
      PADDLE_ENFORCE_EQ(velocity_list[i],
                        velocity_out_list[i],
                        platform::errors::InvalidArgument(
                            "Input(Velocity) and Output(VelocityOut) must be "
                            "the same Tensors."));
      if (multi_precision) {
        master_param_list.push_back(
            const_cast<float*>(master_param[i]->data<float>()));
        master_param_out_list.push_back(
            master_param_out[i]->mutable_data<float>(ctx.GetPlace()));
        PADDLE_ENFORCE_EQ(master_param_list[i],
                          master_param_out_list[i],
                          platform::errors::InvalidArgument(
                              "Input(MasterParam) and Output(MasterParamOut) "
                              "must be the same Tensors."));
      } else {
        master_param_list.push_back(nullptr);
        master_param_out_list.push_back(nullptr);
      }
    }

    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    int r = lars_momentum(dev_ctx.x_context(),
                          param_list,
                          grad_list,
                          velocity_list,
                          lrs,
                          master_param_list,
                          param_out_list,
                          velocity_out_list,
                          master_param_out_list,
                          weight_decay_arr,
                          param_sizes,
                          mu,
                          lars_coeff,
                          epsilon,
                          rescale_grad);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "lars_momentum");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(lars_momentum,
                       ops::LarsMomentumOpXPUKernel<paddle::platform::float16>,
                       ops::LarsMomentumOpXPUKernel<float>);
#endif
