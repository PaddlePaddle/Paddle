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

#include "paddle/phi/kernels/embedding_bag_grad_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/embedding_util.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
struct EmbeddingBagGradCPUFunctor{
  EmbeddingBagGradCPUFunctor(const Context& dev_ctx,
                              const DenseTensor& input,
                              const DenseTensor& params,
                              const DenseTensor& weight,
                              const DenseTensor& out_grad,
                              const std::string& mode,
                              DenseTensor* params_grad, 
                              DenseTensor* weight_grad) 
                               
      : dev_ctx_(dev_ctx),
        input_(input),
        params_(params),
        weight_(weight),
        out_grad_(out_grad),
        mode_(mode),
        params_grad_(params_grad),
        weight_grad_(weight_grad){}

    using EigenVectorMap = Eigen::Map< Eigen::Vector<T, Eigen::Dynamic> >;
    using ConstEigenVectorMap = Eigen::Map< const Eigen::Vector<T, Eigen::Dynamic> >;
    using EigenIndex = Eigen::Index;


  template <typename IdT>
  void apply() {
    dev_ctx_.template Alloc<T>(params_grad_);
    dev_ctx_.template Alloc<T>(weight_grad_);
    
    const EigenIndex sequence_length = input_.dims()[1];
    const EigenIndex output_dim = params_.dims()[1];

    std::unordered_map<IdT, EigenIndex> index_map;
    std::vector< std::pair<IdT, std::vector<EigenIndex>>> index_vec;

    auto ids = CopyIdsToVector<IdT, int64_t>(input_);

    auto* d_grad = out_grad_.data<T>();
    auto* d_weights = weight_.data<T>();
    auto* d_params = params_.data<T>();
    auto* d_inputs = input_.data<IdT>();
    
    auto* d_params_grad = params_grad_->data<T>();
    auto* d_weight_grad = weight_grad_->data<T>();
    
    auto ids_num = static_cast<int64_t>(ids.size());

    for (EigenIndex i = 0; i < ids_num ; ++i) {
      auto index = ids.data()[i];
      if (index_map.find(index) == index_map.end()) {
        index_map[index] = index_vec.size();
        index_vec.push_back({index,{}});
      }
      index_vec[index_map[index]].second.push_back(i);
    }

    EigenIndex bags = input_.dims()[0];
    for (EigenIndex i = 0; i < bags; ++i) {
      EigenVectorMap params_grads_slice(&d_params_grad[index_vec[i].first * output_dim], output_dim );
      
      for (EigenIndex index : index_vec[i].second) {
        const EigenIndex bag = index / sequence_length;
        const EigenIndex seq = index % sequence_length;
        const ConstEigenVectorMap grads_slice(&d_grad[bag*output_dim], output_dim);
        params_grads_slice += grads_slice * d_weights[bag*sequence_length + seq];
      }
      if (mode_ == "mean") {
        params_grads_slice /= static_cast<T>(sequence_length);
      }
    }

    for (EigenIndex i=0; i<bags; ++i){

      for (EigenIndex j=0; j<sequence_length; ++j){
        const ConstEigenVectorMap grads_slice( &d_grad[i * output_dim ], output_dim );
        const ConstEigenVectorMap params_slice(&d_params[d_inputs[i*sequence_length+j] * output_dim ]
                                                , output_dim );
        if (mode_ == "sum"){
          d_weight_grad[i * sequence_length + j]  =   params_slice.dot(grads_slice);
        }else {
          d_weight_grad[i * sequence_length + j] = params_slice.dot(grads_slice) / 
                      static_cast<T>(sequence_length);
        }
        
      }
    }
  } 

    private:
      const Context& dev_ctx_;
      const DenseTensor& input_;
      const DenseTensor& params_;
      const DenseTensor& weight_;
      const DenseTensor& out_grad_;
      const std::string& mode_;
      DenseTensor* params_grad_;
      DenseTensor* weight_grad_;
      

};

template <typename T, typename Context>
void EmbeddingBagGradKernel(const Context& ctx,
                            const DenseTensor& input,
                            const DenseTensor& params,
                            const DenseTensor& weight,
                            const DenseTensor& out_grad,
                            const std::string& mode,
                            DenseTensor* params_grad,
                            DenseTensor* weight_grad) {
  EmbeddingBagGradCPUFunctor<T, Context> functor(ctx, input, params, weight, out_grad, mode, params_grad, weight_grad);

  if (input.dtype() == phi::DataType::INT32) 
  {
    functor.template apply<int>();
  } else if (input.dtype() == phi::DataType::INT64) 
  {
    functor.template apply<int64_t>();
  } else {
      PADDLE_THROW(phi::errors::Unimplemented("emebdding input only support int32 and int64"));
  }

}

    
}  // namespace phi

PD_REGISTER_KERNEL(embedding_bag_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::EmbeddingBagGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}