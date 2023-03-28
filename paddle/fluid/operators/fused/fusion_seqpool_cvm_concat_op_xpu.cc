/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include <string>
#include <vector>
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/operators/fused/fusion_seqpool_cvm_concat_op.h"

namespace paddle {
namespace operators {
using Tensor = phi::DenseTensor;
using LoDTensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class FusionSeqPoolCVMConcatGradXPUKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* dOut =
          ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    const Tensor* cvm = ctx.Input<Tensor>("CVM");
    auto dxs = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));
    auto use_cvm = ctx.Attr<bool>("use_cvm");//TODO:
    int n = dxs.size();
    auto xpu_context =
        ctx.template device_context<DeviceContext>().x_context();
    // auto place = ctx.GetPlace();
    phi::Place l3_place = ctx.template device_context<DeviceContext>().GetL3Place();
    T* dy_data = const_cast<T*>(dOut->data<T>());
    T* cvm_data = const_cast<T*>(cvm->data<T>());
    int batch_size = dOut->dims()[0];
    int dy_offset = dOut->dims()[1];
    auto item_size = dxs[0]->dims()[1];
    std::vector<const T*> cpu_dx_list(n);
    unsigned int sum_size = n * (batch_size + 1);
    std::vector<int> cpu_lodx(sum_size);
    unsigned int start_index = 0;
    for (int k = 0; k < n; k++) {
        auto dx = dxs[k];
        const T* dx_data = dx->mutable_data<T>(l3_place);
        auto lod = dx->lod();
        cpu_dx_list[k] = dx_data;
        auto lod_level_0 = dx->lod()[0];
        int lod_size = lod_level_0.size();
        for (int i = 0; i < lod_size; i++) {
	    cpu_lodx[i + start_index] = lod_level_0[i];
	}
	start_index += lod_size;
    }
    int r = xpu::sequence_sum_pool_cvm_concat_grad<T>(xpu_context, dy_data, cvm_data,
                               cpu_dx_list, cpu_lodx,
                               item_size, batch_size, n, dy_offset, use_cvm);
     PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
            platform::errors::External(
               "The sequence_pool_cvm_grad XPU OP return wrong value[%d %s]",
               r, XPUAPIErrorMsg[r]));
  }
};

template <typename DeviceContext, typename T>
class FusionSeqPoolCVMConcatXPUKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");
    std::string pooltype = ctx.Attr<std::string>("pooltype");
    bool use_cvm = ctx.Attr<bool>("use_cvm");
    auto x0_lod = ins[0]->lod();
    auto x0_dims = ins[0]->dims();
    auto y_dims = out->dims();
    auto xpu_context =
        ctx.template device_context<DeviceContext>().x_context();
    size_t bs = x0_lod[0].size() - 1;
    out->Resize({static_cast<int64_t>(bs), y_dims[1]});
    framework::LoD y_lod(1);
    y_lod[0].resize(bs + 1);
    for (size_t i = 0; i <= bs; ++i) {
        y_lod[0][i] = i;
    }
    out->set_lod(y_lod);
    // auto place = ctx.GetPlace();
    phi::Place l3_place = ctx.template device_context<DeviceContext>().GetL3Place();
    T* y_data = out->mutable_data<T>(l3_place);
    int w = ins[0]->numel() / x0_dims[0];
    if(use_cvm) {
      PADDLE_ENFORCE_EQ(y_dims[1] % w, 0,
                        paddle::platform::errors::InvalidArgument(
                            "The output of dims[1] should be dividable of w"));
    }
    else{
      PADDLE_ENFORCE_EQ(y_dims[1] % (w-2), 0,
                  paddle::platform::errors::InvalidArgument(
                      "The output of dims[1] should be dividable of (w-2)"));
    }
    int slot_num = static_cast<int>(ins.size());
    std::vector<const T*> cpu_x_addr_vec(slot_num, 0);
    unsigned int sum_lod_size = slot_num * (bs + 1);
    std::vector<int> cpu_lodx(sum_lod_size);
    unsigned int lod_index = 0;
    for (int i = 0; i < slot_num; i++) {
        cpu_x_addr_vec[i] = reinterpret_cast<const T*>(ins[i]->data<T>());
        auto x_lod = ins[i]->lod()[0];
        for (size_t j = 0; j < x_lod.size(); j++) {
           cpu_lodx[lod_index + j] = x_lod[j];
        }
	lod_index += x_lod.size();
    }
    int r = xpu::sequence_sum_pool_cvm_concat<T>(xpu_context, cpu_x_addr_vec, y_data, cpu_lodx,
                    bs, x0_dims[1], 0.00f, slot_num, use_cvm);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                     platform::errors::External(
                         "The sequence_sum_pool XPU OP return wrong value[%d %s]",
                         r, XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    fusion_seqpool_cvm_concat,
    ops::FusionSeqPoolCVMConcatXPUKernel<paddle::platform::XPUDeviceContext, float>);

REGISTER_OP_XPU_KERNEL(
    fusion_seqpool_cvm_concat_grad,
    ops::FusionSeqPoolCVMConcatGradXPUKernel<paddle::platform::XPUDeviceContext, float>);