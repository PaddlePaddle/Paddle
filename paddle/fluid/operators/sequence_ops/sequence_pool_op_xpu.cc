#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/operators/sequence_ops/sequence_pool_op.h"
namespace paddle {
    namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class SequencePoolGradXPUKernel : public framework::OpKernel<T> {
    public:
        void Compute(const framework::ExecutionContext& context) const override {
            auto* out_g = context.Input<LoDTensor>(framework::GradVarName("Out"));
            auto* in_g = context.Output<LoDTensor>(framework::GradVarName("X"));
            std::string pooltype = context.Attr<std::string>("pooltype");
       const Tensor* index = nullptr;
            if (pooltype == "MAX") {
      index = context.Input<Tensor>("MaxIndex");
            }
            in_g->mutable_data<T>(context.GetPlace());
            auto* in = context.Input<LoDTensor>("X");
            PADDLE_ENFORCE_EQ(
               in->lod().empty(), false,
               platform::errors::InvalidArgument("Input(X) Tensor of SequenceConvOp "
                                                 "does not contain LoD information."));
            auto lod = in->lod();
            auto lod_level = lod.size();
            if (lod_level > 1UL) {
                framework::LoD out_lod;
                out_lod.push_back(lod[0]);
            }
            auto dims = in->dims();
            dims[0] = lod[lod_level - 1].size() - 1;
            auto& dev_ctx = context.template device_context<DeviceContext>();
            int r = xpu::Error_t::SUCCESS;
            auto lod_level_0 = in->lod()[0];
            int lod_size = lod_level_0.size();
            int64_t batch_size = static_cast<int64_t>(lod[lod_level - 1].size() - 1);
            std::vector<int> cpu_lodx(lod_size);
            for (int i = 0; i < lod_size; i++) {
               cpu_lodx[i] = lod_level_0[i];
            }
            const xpu::VectorParam<int> lodx = {cpu_lodx.data(),
                  static_cast<int>(cpu_lodx.size()), nullptr};
       xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
       auto lodx_xpu = lodx.to_xpu(RAII_GUARD);
            if (pooltype == "MAX") {
                r = xpu::sequence_max_pool_grad<T, int>(dev_ctx.x_context(),
                       out_g->data<T>(), lodx_xpu.xpu, in_g->data<T>(), batch_size, dims[1], index->data<int>()
                );
                PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                     platform::errors::External(
                         "The sequence_max_pool_grad XPU OP return wrong value[%d %s]",
                         r, XPUAPIErrorMsg[r]));
            } else if (pooltype == "SUM") {
                r = xpu::sequence_sum_pool_grad<T, int>(dev_ctx.x_context(), out_g->data<T>(), lodx_xpu.xpu, in_g->data<T>(),
                    batch_size, dims[1]);
                PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                    platform::errors::External("The sequence_sum_pool_grad XPU OP return wrong value[%d %s]", r, XPUAPIErrorMsg[r]));
            } else if (pooltype == "LAST") {
                r = xpu::sequence_last_pool_grad<T, int>(dev_ctx.x_context(),  out_g->data<T>(), lodx_xpu.xpu, in_g->data<T>(),
                    batch_size, dims[1]);
                PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                    platform::errors::External("The sequence_last_pool_grad XPU OP return wrong value[%d %s]", r, XPUAPIErrorMsg[r]));
            } else if (pooltype == "FIRST") {
                r = xpu::sequence_first_pool_grad<T, int>(dev_ctx.x_context(), out_g->data<T>(), lodx_xpu.xpu, in_g->data<T>(),
                     batch_size, dims[1]);
                PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                    platform::errors::External("The sequence_first_pool_grad XPU OP return wrong value[%d %s]", r, XPUAPIErrorMsg[r]));
            } else {
            }
   }
};

template <typename DeviceContext, typename T>
class SequencePoolXPUKernel : public framework::OpKernel<T> {
    public:
        void Compute(const framework::ExecutionContext& context) const override {
            auto* in = context.Input<LoDTensor>("X");
            auto* out = context.Output<LoDTensor>("Out");
            out->mutable_data<T>(context.GetPlace());
            std::string pooltype = context.Attr<std::string>("pooltype");
            T pad_value = static_cast<T>(context.Attr<float>("pad_value"));
            PADDLE_ENFORCE_EQ(
               in->lod().empty(), false,
               platform::errors::InvalidArgument("Input(X) Tensor of SequenceConvOp "
                                                 "does not contain LoD information."));
            auto lod = in->lod();
            auto lod_level = lod.size();
            if (lod_level > 1UL) {
                framework::LoD out_lod;
                out_lod.push_back(lod[0]);
                out->set_lod(out_lod);
            }
            Tensor* index = nullptr;
            auto dims = in->dims();
            dims[0] = lod[lod_level - 1].size() - 1;
            //int64_t in_w = in->numel() / dims[0];
            out->Resize({dims});
            out->mutable_data<T>(context.GetPlace());
            auto& dev_ctx = context.template device_context<DeviceContext>();
            int r = xpu::Error_t::SUCCESS;
            auto lod_level_0 = in->lod()[0];
            int lod_size = lod_level_0.size();
            int64_t batch_size = static_cast<int64_t>(lod[lod_level - 1].size() - 1);
            std::vector<int> cpu_lodx(lod_size);
            for (int i = 0; i < lod_size; i++) {
                cpu_lodx[i] = lod_level_0[i];
            }
            const xpu::VectorParam<int> lodx = {cpu_lodx.data(),
                             static_cast<int>(cpu_lodx.size()), nullptr};
            if (pooltype == "MAX") {
                index = context.Output<Tensor>("MaxIndex");
                index->Resize({dims});
                index->mutable_data<int>(context.GetPlace());
                r = xpu::sequence_max_pool<T, int>(dev_ctx.x_context(),
                        in->data<T>(),  out->data<T>(),  lodx, batch_size, in->numel()/batch_size, pad_value, index->data<int>()
                );
                PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                     platform::errors::External(
                         "The sequence_max_pool XPU OP return wrong value[%d %s]",
                         r, XPUAPIErrorMsg[r]));
            } else if (pooltype == "SUM") {
                 // template<typename T, typename TID> int sequence_sum_pool(Context* ctx, const T* x, T* y, const VectorParam<TID>& lod,
                 // int batch, int dim, float pad_value) ;
                 r = xpu::sequence_sum_pool<T, int>(dev_ctx.x_context(),  in->data<T>(),  out->data<T>(), lodx,
                   batch_size, dims[1], pad_value);
                PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                    platform::errors::External("The sequence_sum_pool XPU OP return wrong value[%d %s]", r, XPUAPIErrorMsg[r]));
            } else if (pooltype == "LAST") {
                r = xpu::sequence_last_pool<T, int>(dev_ctx.x_context(),  in->data<T>(),  out->data<T>(), lodx,
                    batch_size, dims[-1], pad_value);
                PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                    platform::errors::External("The sequence_last_pool XPU OP return wrong value[%d %s]", r, XPUAPIErrorMsg[r]));
            } else if (pooltype == "FIRST") {
                r = xpu::sequence_first_pool<T, int>(dev_ctx.x_context(),  in->data<T>(),  out->data<T>(), lodx,
                    batch_size, dims[-1], pad_value);
                PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                     platform::errors::External("The sequence_first_pool XPU OP return wrong value[%d %s]", r, XPUAPIErrorMsg[r]));
            } else {
            }
        }
};
    }  // namespace operators
} // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    sequence_pool, ops::SequencePoolXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    sequence_pool_grad, 
        ops::SequencePoolGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif

