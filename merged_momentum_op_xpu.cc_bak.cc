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
    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    auto mu_ptr = &mu;
    //std::cout<<"this is T"<<T;
    //auto& xpu_ctx =ctx.template device_context<paddle::platform::XPUDeviceContext>();
    auto params = ctx.MultiInput<framework::Tensor>("Param");
    std::cout<<"params dress. ******************"<<params[0]<<std::endl;
    auto params_out = ctx.MultiOutput<framework::Tensor>("ParamOut");
    auto lr = ctx.Input<framework::Tensor>("LearningRate");
    size_t n = params.size();
    auto velocity = ctx.MultiInput<framework::Tensor>("Velocity");
    auto grad = ctx.MultiInput<framework::Tensor>("Grad");
    auto velocity_out = ctx.MultiOutput<framework::Tensor>("VelocityOut");
    auto use_nesterov = ctx.Attr<bool>("use_nesterov");
    int sum=0;
    std::vector<int> sizes(n);
    if(n>0){
       for(size_t j=0;j<n;j++){
           sizes[j]=params[j]->numel();
           sum +=sizes[j];	   
      }
    }
  
    std::cout<<"sum is  ******************"<<sum<<"this is n "<<n<<std::endl;
    std::cout<<"1. ******************"<<std::endl;
    
    paddle::framework::Tensor params_vec;
    params_vec.Resize({sum});
    paddle::framework::Tensor params_out_vec;
    params_out_vec.Resize({sum});
    paddle::framework::Tensor grad_vec;
    grad_vec.Resize({sum});
    //paddle::framework::Tensor tmp_vec;
    paddle::framework::Tensor velocity_vec;
    velocity_vec.Resize({sum});
    paddle::framework::Tensor velocity_out_vec;
    velocity_out_vec.Resize({sum});
    
    //std::vector 
 
    std::cout<<"2. ******************"<<std::endl;
    
    for(size_t i=0;i<n;i++){
      if(params[i] && params[i]->numel()>0){
        int idx=0;
	std::cout<<"111 params. ******************"<<std::endl;
        auto* keep_data = params_vec.mutable_data<T>(ctx.GetPlace());
        std::cout<<"2222 params. ******************"<<std::endl;	
        auto* y_data = params[i]->data<T>();       	        
        std::cout<<"3333 params. ******************"<<std::endl;	
	for(int j=0;j<params[i]->numel();j++){
           std::cout<<"444 params. ******************"<<std::endl;	
	   keep_data[j+idx]=y_data[j];
      } 
	idx +=params[i]->numel();
        //std::cout<<"4. ******************"<<params_out_vec[1]<<std::endl;
        std::cout<<"4. ******************"<<std::endl;
      }
      if(params_out[i] && params_out[i]->numel()>0){
        int idx=0;
        auto* keep_data = params_out_vec.mutable_data<T>(ctx.GetPlace());       	
        auto* y_data = params_out[i]->data<T>();       	
        for(int j=0;j<params_out[i]->numel();j++){
	   keep_data[j+idx]=y_data[j];
      } 
	idx +=params_out[i]->numel();
        std::cout<<"5. ******************"<<std::endl;
      }     
      if(grad[i] && grad[i]->numel()>0){
        int idx=0;
        auto* keep_data = grad_vec.mutable_data<T>(ctx.GetPlace());       	
        auto* y_data = grad[i]->data<T>();       	
        for(int j=0;j<grad[i]->numel();j++){
	   keep_data[j+idx]=y_data[j];
      } 
	idx +=grad[i]->numel();
        std::cout<<"6. ******************"<<std::endl;
      }     
      if(velocity[i] && velocity[i]->numel()>0){
        int idx=0;
        auto* keep_data = velocity_vec.mutable_data<T>(ctx.GetPlace());       	
        auto* y_data = velocity[i]->data<T>();       	
        for(int j=0;j<velocity[i]->numel();j++){
	   keep_data[j+idx]=y_data[j];
      } 
	idx +=velocity[i]->numel();
        std::cout<<"8. ******************"<<std::endl;
      }     
      if(velocity_out[i] && velocity_out[i]->numel()>0){
        int idx=0;
        auto* keep_data = velocity_out_vec.mutable_data<T>(ctx.GetPlace());       	
        auto* y_data = velocity_out[i]->data<T>();       	
        for(int j=0;j<velocity_out[i]->numel();j++){
	   keep_data[j+idx]=y_data[j];
      } 
	idx +=velocity_out[i]->numel();
        std::cout<<"9. ******************"<<std::endl;
      }     
    }
    /*
     for (int i=0;i<100;i++){
        std::cout<<"param_vec. ******************"<<params_vec[i]<<std::endl;
     }
     
    auto* params_ptr = VectorToTensor<T>(params_vec,ctx,static_cast<int>(params_out_vec.size()));
    //std::cout<<"params_tensor ******************"<<params_ptr<<"$$$$$"<<*(params_ptr)<<std::endl;
    auto* params_out_ptr = VectorToTensor<T>(params_out_vec,ctx,static_cast<int>(params_out_vec.size()));
    //std::cout<<"params_out_tensor ******************"<<*params_out_ptr<<std::endl;
    auto* grad_ptr = reinterpret_cast<T*>(VectorToTensor<T>(grad_vec,ctx,static_cast<int>(params_out_vec.size())));
    //std::cout<<"grad_tensor ******************"<<*grad_ptr<<std::endl;
    auto* lr_ptr = reinterpret_cast<float*>(VectorToTensor<float>(lr_vec,ctx,static_cast<int>(params_out_vec.size())));
    //std::cout<<"lr_tensor ******************"<<*lr_ptr<<std::endl;
    auto* velocity_ptr = reinterpret_cast<T*>(VectorToTensor<T>(velocity_vec,ctx,static_cast<int>(params_out_vec.size())));
    //std::cout<<"ve_tensor ******************"<<*velocity_ptr<<std::endl;
    auto* velocity_out_ptr = VectorToTensor<T>(velocity_out_vec,ctx,static_cast<int>(params_out_vec.size()));
    //std::cout<<"ve_out_tensor ******************"<<*velocity_out_ptr<<std::endl;
    */
    
    auto* sizes_ptr = reinterpret_cast<int*>(VectorToTensor<int>(sizes,ctx,static_cast<int>(sizes.size())));    
    //std::cout<<"sizes_tensor ******************"<<*sizes_ptr<<std::endl;
    
    
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    std::cout<<"108. ******************"<<std::endl;
    int len =static_cast<int>(n);
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
    std::cout<<"12. ******************"<<std::endl;
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
    
    std::cout<<"13. ******************"<<std::endl;
    int r = xpu::merged_momentum(dev_ctx.x_context(),params_vec.data<T>(),
                          velocity_vec.data<T>(),grad_vec.data<T>(),
                          params_out_vec.data<T>(),velocity_out_vec.data<T>(),
                          sum,lr->data<T>(), use_nesterov,mu_ptr,len,sizes_ptr);
    std::cout<<"14. ******************"<<std::endl;
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
  }
};
}
}

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    merged_momentum,
    ops::MergedMomentumOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif


