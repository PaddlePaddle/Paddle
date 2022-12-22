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
    int n = dxs.size();
    auto xpu_context =
        ctx.template device_context<DeviceContext>().x_context();
    T* dy_data = const_cast<T*>(dOut->data<T>());
    T* cvm_data = const_cast<T*>(cvm->data<T>());
    int batch_size = dOut->dims()[0];
    int dy_offset = dOut->dims()[1];
    auto item_size = dxs[0]->dims()[1];
    std::unique_ptr<unsigned long long[]> cpu_dx_list(new unsigned long long[n]);
    std::unique_ptr<unsigned long long[]> xpu_lod_list(new unsigned long long[n]);
    xpu::VectorParam<unsigned long long> dx_vec_param = {&cpu_dx_list[0], n, nullptr};
    xpu::VectorParam<unsigned long long> lod_vec_param = {nullptr, n, nullptr};
    unsigned int sum_size = 0;
    for (int k = 0; k < n; k++) {
        auto dx = dxs[k];
        auto lod_level_0 = dx->lod()[0];
        int lod_size = lod_level_0.size();
	sum_size += lod_size;
    }
    xpu::ctx_guard RAII_GUARD(xpu_context);
    unsigned long long lod_addr = reinterpret_cast<unsigned long long>(RAII_GUARD.alloc_l3_or_gm<unsigned int>(sum_size));
    std::vector<unsigned int> cpu_lodx(sum_size);
    unsigned int start_index = 0;
    for (int k = 0; k < n; k++) {
        auto dx = dxs[k];
        // T* dx_data = dx->mutable_data_l3<T>(ctx.GetPlace());
        T* dx_data = dx->mutable_data<T>(ctx.GetPlace());
        auto lod = dx->lod();
        cpu_dx_list[k] = reinterpret_cast<unsigned long long>(dx_data);
        auto lod_level_0 = dx->lod()[0];
        int lod_size = lod_level_0.size();
	xpu_lod_list[k] = lod_addr + start_index * sizeof(unsigned int);
        for (int i = 0; i < lod_size; i++) {
	    cpu_lodx[i + start_index] = lod_level_0[i];
	}
	start_index += lod_size;
    }
    xpu_memcpy(reinterpret_cast<void*>(lod_addr), cpu_lodx.data(), sum_size * sizeof(unsigned int), XPU_HOST_TO_DEVICE);
    dx_vec_param.xpu = reinterpret_cast<unsigned long long*>(RAII_GUARD.alloc_l3_or_gm<unsigned long long>(n));
    xpu_memcpy(dx_vec_param.xpu, &cpu_dx_list[0], n * sizeof(uint64_t), XPU_HOST_TO_DEVICE);
    lod_vec_param.xpu = reinterpret_cast<unsigned long long*>(RAII_GUARD.alloc_l3_or_gm<unsigned long long>(n));
    xpu_memcpy(lod_vec_param.xpu, &xpu_lod_list[0], n * sizeof(unsigned long long), XPU_HOST_TO_DEVICE);
    int r = xpu::sequence_sum_pool_cvm_grad<T>(xpu_context, dy_data, cvm_data,
                               dx_vec_param, lod_vec_param,
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
    auto place = ctx.GetPlace();
    T* y_data = out->mutable_data<T>(place);
    // T* y_data = out->mutable_data_l3<T>(place);
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
    unsigned int sum_lod_size = 0;
    unsigned int lod_index = 0;
    for (int i = 0; i < slot_num; i++) {
        auto x_lod = ins[i]->lod()[0];
	sum_lod_size += x_lod.size();
    }
    int* xpu_lod_ptr = RAII_GUARD.alloc_l3_or_gm<int>(sum_lod_size);
    std::vector<int> cpu_lodx(sum_lod_size);

    for (int i = 0; i < slot_num; i++) {
        xpu_x_addr_vec[i] = reinterpret_cast<uint64_t>(ins[i]->data<T>());
        auto x_lod = ins[i]->lod()[0];
        for (size_t j = 0; j < x_lod.size(); j++) {
           cpu_lodx[lod_index + j] = x_lod[j];
        }
        xpu_lod_addr_vec[i] = reinterpret_cast<uint64_t>(xpu_lod_ptr + lod_index);
	lod_index += x_lod.size();
    }
    xpu_memcpy(xpu_lod_ptr, cpu_lodx.data(), sum_lod_size * sizeof(int), XPU_HOST_TO_DEVICE);
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
