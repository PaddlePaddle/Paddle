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
#ifdef PADDLE_WITH_XPU
#include <iostream>
#include <unistd.h>
#include <sys/syscall.h>
#include "paddle/fluid/operators/optimizers/merged_momentum_op.h"
namespace paddle {
namespace operators {

template <typename T>
inline  T* VectorToTensor(const std::vector<T>& selected_indices,const framework::ExecutionContext& ctx, int selected_num) {
        paddle::framework::Tensor keep_nms;
        keep_nms.Resize({selected_num});
        //auto* keep_data = keep_nms.mutable_data<T>(ctx.GetPlace());
        auto* keep_data = keep_nms.mutable_data<T>(platform::CPUPlace());
        for (int i = 0; i < selected_num; ++i) {
           keep_data[i] = selected_indices[i];
           std::cout<<"keep_nms element:"<<keep_data[i]<<std::endl;
        }
        return keep_nms.data<T>();
     }


template <typename DeviceContext, typename T>
class MergedMomentumOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    //T mu = ctx.Attr<float>("mu");
    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    std::vector<T> mu_vec;
    //T* mu_ptr = reinterpret_cast<T*>(&mu_vec);
    //std::cout<<"this is T"<<T;
    //auto& xpu_ctx =ctx.template device_context<paddle::platform::XPUDeviceContext>();
    auto params = ctx.MultiInput<framework::Tensor>("Param");
    //std::cout<<*params[0]<<std::endl;
    //std::cout<<"params dress. ******************"<<params[0]<<std::endl;
    auto params_out = ctx.MultiOutput<framework::Tensor>("ParamOut");
    auto lr = ctx.Input<framework::Tensor>("LearningRate");
    size_t n = params.size();
    int len =static_cast<int>(n);
    auto velocity = ctx.MultiInput<framework::Tensor>("Velocity");
    auto grad = ctx.MultiInput<framework::Tensor>("Grad");
    auto velocity_out = ctx.MultiOutput<framework::Tensor>("VelocityOut");
    auto use_nesterov = ctx.Attr<bool>("use_nesterov");
    int sum=0;
    std::vector<int> sizes(n);
    if(n>0){
       for(size_t j=0;j<n;j++){
           sizes[j]=params[j]->numel();
           //std::cout<<"params: "<<params[j]->numel()<<" params_out:"<<params_out[j]->numel()<<" grad"<<grad[j]->numel()<<" velocity:"<<velocity[j]->numel()<<std::endl;  
           sum +=sizes[j];
	   mu_vec.push_back(mu);
      }
    }


    T* params_ptr = nullptr;
    T* params_out_ptr = nullptr;
    T* grad_ptr = nullptr;
    T* velocity_ptr = nullptr;
    T* velocity_out_ptr = nullptr;
    int * sizes_xpu_ptr = nullptr;
    float * mu_xpu_ptr = nullptr;
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&params_ptr),
                   sum),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted(
            "\n\nOut of memory error on XPU, Cannot allocate %s memory"
            " on XPU. \n\nPlease check whether there is any other process "
            "using XPU.\n",
            string::HumanReadableSize(sum)));
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&params_out_ptr),
                   sum*sizeof(T)),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted(
            "\n\nOut of memory error on XPU, Cannot allocate %s memory"
            " on XPU. \n\nPlease check whether there is any other process "
            "using XPU.\n",
            string::HumanReadableSize(sum)));
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&grad_ptr),
                   sum*sizeof(T)),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted(
            "\n\nOut of memory error on XPU, Cannot allocate %s memory"
            " on XPU. \n\nPlease check whether there is any other process "
            "using XPU.\n",
            string::HumanReadableSize(sum)));
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&velocity_ptr),
                   sum*sizeof(T)),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted(
            "\n\nOut of memory error on XPU, Cannot allocate %s memory"
 " on XPU. \n\nPlease check whether there is any other process "
            "using XPU.\n",
            string::HumanReadableSize(sum)));
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&velocity_out_ptr),
                   sum*sizeof(T)),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted(
            "\n\nOut of memory error on XPU, Cannot allocate %s memory"
            " on XPU. \n\nPlease check whether there is any other process "
            "using XPU.\n",
            string::HumanReadableSize(sum)));
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&sizes_xpu_ptr),n*sizeof(int)),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted(
            "\n\nOut of memory error on XPU, Cannot allocate %s memory"
            " on XPU. \n\nPlease check whether there is any other process "
            "using XPU.\n",
            string::HumanReadableSize(len)));
    PADDLE_ENFORCE_EQ(
        xpu_malloc(reinterpret_cast<void**>(&mu_xpu_ptr),n*sizeof(int)),
        XPU_SUCCESS,
        platform::errors::ResourceExhausted(
            "\n\nOut of memory error on XPU, Cannot allocate %s memory"
            " on XPU. \n\nPlease check whether there is any other process "
            "using XPU.\n",
            string::HumanReadableSize(len)));

    int idx=0;
    for(size_t i=0;i<n;i++){
      if(params[i] && params[i]->numel()>0){ 
        memory::Copy(ctx.GetPlace(),params_ptr+idx,ctx.GetPlace(),params[i]->data<T>(), (params[i]->numel())*sizeof(T));
      }
      if(params_out[i] && params_out[i]->numel()>0){ 
        memory::Copy(ctx.GetPlace(),params_out_ptr+idx,ctx.GetPlace(),params_out[i]->data<T>(), (params_out[i]->numel())*sizeof(T));
      }
      if(grad[i] && grad[i]->numel()>0){ 
        memory::Copy(ctx.GetPlace(),grad_ptr+idx,ctx.GetPlace(),grad[i]->data<T>(), (grad[i]->numel())*sizeof(T));
      }
      if(velocity[i] && velocity[i]->numel()>0){ 
        memory::Copy(ctx.GetPlace(),velocity_ptr+idx,ctx.GetPlace(),velocity[i]->data<T>(), (velocity[i]->numel())*sizeof(T));
      }
      if(velocity_out[i] && velocity_out[i]->numel()>0){ 
        memory::Copy(ctx.GetPlace(),velocity_out_ptr+idx,ctx.GetPlace(),velocity_out[i]->data<T>(), (velocity_out[i]->numel())*sizeof(T));
      }
      idx +=params[i]->numel();
    }
    memory::Copy(ctx.GetPlace(),sizes_xpu_ptr, platform::CPUPlace(), &sizes, n*sizeof(int));
    memory::Copy(ctx.GetPlace(),mu_xpu_ptr, platform::CPUPlace(), &mu_vec, n*sizeof(float));
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    PADDLE_ENFORCE_EQ(n, params_out.size(),
                      platform::errors::InvalidArgument(
                          "The size of Output(ParamOut) must be equal to "
                          "Input(Param), but got the size of Output(ParamOut) "
                          "is %d, the size of Input(Param) is %d.",
                          params_out.size(), n));
    PADDLE_ENFORCE_EQ(n, velocity.size(),
                      platform::errors::InvalidArgument(
                          "The size of Output(ParamOut) must be equal to "
"Input(Param), but got the size of Output(ParamOut) "
                          "is %d, the size of Input(Param) is %d.",
                          velocity.size(), n));
    PADDLE_ENFORCE_EQ(n, velocity_out.size(),
                      platform::errors::InvalidArgument(
                          "The size of Output(ParamOut) must be equal to "
                          "Input(Param), but got the size of Output(ParamOut) "
                          "is %d, the size of Input(Param) is %d.",
                          velocity_out.size(), n));
    for (size_t i = 0; i < n; ++i) {
      PADDLE_ENFORCE_EQ(params[i], params_out[i],
                        platform::errors::InvalidArgument(
                            "The size of Input(Param) and Output(ParamOut) "
                            "must be the same Tensors."));
    }
    PADDLE_ENFORCE_EQ(
        n, grad.size(),
        platform::errors::InvalidArgument(
            "The size of Input(Grad) must be equal to Input(Param), but got "
            "the size of Input(Grad) is %d, the size of Input(Param) is %d.",
            grad.size(), n));
    std::cout<<"len :"<<len<<"sum :"<<sum<<std::endl;
    int r = xpu::merged_momentum(dev_ctx.x_context(),params_ptr,
                          velocity_ptr,grad_ptr,
                          params_out_ptr,velocity_out_ptr,
                          sum,lr->data<T>(), use_nesterov,mu_xpu_ptr,len,sizes_xpu_ptr);
    int tid = syscall(SYS_gettid);
    std::cout<<"14. merged momentum:" << dev_ctx.x_context() << " " << dev_ctx.x_context()->xpu_stream <<  " "
	                              << (int) dev_ctx.x_context()->dev().type() << " " << dev_ctx.x_context()->dev().id() << " tid:" << tid << std::endl;
    
    r = xpu_wait();  
    PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::InvalidArgument(
              "XPU kernel error of MomentumOp, error message: INVALID_PARAM, "
              "please check your input & output."));
    if (r == xpu::Error_t::INVALID_PARAM) {
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::InvalidArgument(
              "XPU kernel error of MomentumOp, error message: INVALID_PARAM, "
              "please check your input & output."));
    } else if (r == xpu::Error_t::RUNTIME_ERROR) {
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::Unavailable(
              "XPU kernel error of MomentumOp, error message: RUNTIME_ERROR, "
              "please check whether Baidu Kunlun card is properly installed."));
    } else if (r == xpu::Error_t::NO_ENOUGH_WORKSPACE) {
      PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                        platform::errors::ResourceExhausted(
                            "XPU kernel error of MomentumOp, error message: "
 "NO_ENOUGH_WORKSPACE, XPU has no enough memory."));
    }
    xpu_free(params_ptr);
    xpu_free(params_out_ptr);
    xpu_free(grad_ptr);
    xpu_free(velocity_ptr);
    xpu_free(velocity_out_ptr);
    xpu_free(sizes_xpu_ptr);
    xpu_free(mu_xpu_ptr);

  }
};
}
}

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    merged_momentum,
    ops::MergedMomentumOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif


