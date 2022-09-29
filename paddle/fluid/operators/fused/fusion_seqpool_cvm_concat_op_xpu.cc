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
template <typename DeviceContext, typename T>
class FusionSeqPoolCVMConcatGradXPUKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* dOut =
          ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    const Tensor* cvm = ctx.Input<Tensor>("CVM");
    auto dxs = ctx.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));
    auto use_cvm = ctx.Attr<bool>("use_cvm");//TODO:
    size_t n = dxs.size();
    auto xpu_context =
          ctx.template device_context<DeviceContext>().x_context();
    auto cvm_offset = use_cvm ? 0 : 2;

    for (size_t k = 0; k < n; k++) {
        auto dx = dxs[k];
        T* dx_data = dx->mutable_data<T>(ctx.GetPlace());
        int tmp_cvm_dx_bs = dOut->dims()[0];
        auto lod = dx->lod();
        auto batch_size = dx->dims()[0];
        dx->mutable_data<T>(ctx.GetPlace());
        const T* cvm_data = cvm->data<T>();
        auto lod_level_0 = dx->lod()[0];
        auto item_size = dx->numel() / batch_size;
        const T* dout_data = dOut->data<T>() + k * (item_size-cvm_offset);
        int lod_size = lod_level_0.size();
        std::vector<unsigned int> cpu_lodx(lod_size);
        for (int i = 0; i < lod_size; i++) {
            cpu_lodx[i] = lod_level_0[i];
        }
        const xpu::VectorParam<unsigned int> lodx = {reinterpret_cast<const unsigned int*>(cpu_lodx.data()),
                    static_cast<int>(cpu_lodx.size()), nullptr};
        xpu::ctx_guard RAII_GUARD(xpu_context);
        auto dy_offset = dOut->dims()[1];
        // use_cvm 0 item_size 11batch_size 4096*2  tmp_cvm_dx_bs 4096dout_offset 765
        auto lodx_xpu = lodx.to_xpu(RAII_GUARD);
        int r = xpu::sequence_sum_pool_cvm_grad<float, unsigned int>(xpu_context, dout_data, dx_data, cvm_data, lodx_xpu.xpu, item_size, tmp_cvm_dx_bs, dy_offset, use_cvm);
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
              platform::errors::External(
                "The sequence_pool_cvm_grad XPU OP return wrong value[%d %s]",
                r, XPUAPIErrorMsg[r]));
    }
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
    auto place = ctx.GetPlace();
    T* y_data = out->mutable_data<T>(place);
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

    xpu::ctx_guard RAII_GUARD(xpu_context);

    int slot_num = static_cast<int>(ins.size());

    xpu::VectorParam<uint64_t> x_addr_VP_list = {nullptr, slot_num, nullptr};
    std::vector<uint64_t> xpu_x_addr_vec(slot_num, 0);

    xpu::VectorParam<uint64_t> lod_VP_list = {nullptr, slot_num, nullptr};
    std::vector<uint64_t> xpu_lod_addr_vec(slot_num, 0);

    for (int i = 0; i < slot_num; i++) {
        xpu_x_addr_vec[i] = reinterpret_cast<uint64_t>(ins[i]->data<T>());

        auto x_lod = ins[i]->lod()[0];
        std::vector<int> cpu_lodx(x_lod.size());
        for (size_t j = 0; j < x_lod.size(); j++) {
           cpu_lodx[j] = x_lod[j];
        }

        int* xpu_lod_ptr = RAII_GUARD.alloc_l3_or_gm<int>(x_lod.size());
        xpu_memcpy(xpu_lod_ptr, cpu_lodx.data(), cpu_lodx.size() * sizeof(int), XPU_HOST_TO_DEVICE);
        xpu_lod_addr_vec[i] = reinterpret_cast<uint64_t>(xpu_lod_ptr);
    }

    uint64_t* xpu_x_addr_ptr = RAII_GUARD.alloc_l3_or_gm<uint64_t>(slot_num);
    xpu_memcpy(xpu_x_addr_ptr, xpu_x_addr_vec.data(), slot_num * sizeof(uint64_t), XPU_HOST_TO_DEVICE);
    x_addr_VP_list.xpu = reinterpret_cast<uint64_t*>(xpu_x_addr_ptr);

    uint64_t* xpu_lod_addr_ptr = RAII_GUARD.alloc_l3_or_gm<uint64_t>(slot_num);
    xpu_memcpy(xpu_lod_addr_ptr, xpu_lod_addr_vec.data(), slot_num * sizeof(uint64_t), XPU_HOST_TO_DEVICE);
    lod_VP_list.xpu = reinterpret_cast<uint64_t*>(xpu_lod_addr_ptr);

    int r = xpu::sequence_sum_pool_cvm<T>(xpu_context, x_addr_VP_list, y_data, lod_VP_list,
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
