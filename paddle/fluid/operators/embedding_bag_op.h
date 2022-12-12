/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {


using LoDTensor = phi::DenseTensor;
using Tensor = phi::DenseTensor;
template<typename T>
using EigenArrayMap = Eigen::Map< Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> >;
template<typename T>
using EigenVectorMap = Eigen::Map< Eigen::Vector<T, Eigen::Dynamic> >;
template<typename T>
using ConstEigenVectorMap = Eigen::Map< const Eigen::Vector<T, Eigen::Dynamic> >;
using EigenIndex = Eigen::Index;

template<typename T>
struct EmbeddingBagCPUFunctor {
    EmbeddingBagCPUFunctor(const framework::ExecutionContext &context,
                           const Tensor *input_t)
        : context_(context), input_t_(input_t) {}
    
    template <typename IdT>
    void apply() {
            auto *params_t = context_.Input<Tensor>("params");
            auto *weight_t = context_.Input<Tensor>("weight");

            auto *output_t = context_.Output<Tensor>("out");

            std::string mode_t = context_.Attr<std::string>("mode");

            const EigenIndex bag_number = input_t_->dims()[0];
            const EigenIndex sequence_length = input_t_->dims()[1];
            const EigenIndex output_dim = params_t->dims()[1];

            auto *indices_d = input_t_->data<IdT>();
            auto *params_d = params_t->data<T>();
            auto *weight_d = weight_t->data<T>();
            
            auto *output_d = output_t -> mutable_data<T>(context_.GetPlace());

            for (EigenIndex bag = 0; bag < bag_number; ++bag) {
                EigenVectorMap<T> output_slice(&output_d[ bag * output_dim], output_dim);
                output_slice.setZero();
                for (EigenIndex seq = 0; seq < sequence_length; ++seq) {
                    const ConstEigenVectorMap<T> params_slice( &params_d[ indices_d[bag*sequence_length+seq] * output_dim ], 
                                                                output_dim);
                    output_slice += params_slice * weight_d[bag*sequence_length + seq];
                }
                if (mode_t == "mean") {
                    output_slice /= static_cast<T>(sequence_length);
                }

            }


    } //apply
    
    private:
        const framework::ExecutionContext &context_;
        const Tensor *input_t_;
};  //struct


template <typename T>
class EmbeddingBagKernel : public framework::OpKernel<T> {
    public:
        void Compute(const framework::ExecutionContext &context) const override {
            const auto *indices = context.Input<Tensor>("input");
            EmbeddingBagCPUFunctor<T> functor(context, indices);
            framework::VisitIntDataType(framework::TransToProtoVarType(indices->dtype()), functor);
        }
};


template <typename T>
struct EmbeddingBagGradCPUFunctor {
    EmbeddingBagGradCPUFunctor(const framework::ExecutionContext &context,
                               const Tensor *input_t )
        : context_(context), input_t_(input_t) {}
    
    template <typename IdT>
    void apply() {
        auto *params_grad_t = context_.Output<Tensor>(framework::GradVarName("params"));
        auto *weight_grad_t = context_.Output<Tensor>(framework::GradVarName("weight"));
        auto *params_value_t = context_.Input<Tensor>("params");
        auto *weight_value_t = context_.Input<Tensor>("weight");
        auto *output_t = context_.Input<Tensor>(framework::GradVarName("out"));

        std::string mode_t = context_.Attr<std::string>("mode");

        auto *indices_d = input_t_->data<IdT>();
        auto *params_value_d = params_value_t->data<T>();
        auto *weight_value_d = weight_value_t->data<T>();

        auto *params_grad_d = params_grad_t->mutable_data<T>(context_.GetPlace());
        auto *weight_grad_d = weight_grad_t->mutable_data<T>(context_.GetPlace());
        auto *output_d = output_t->data<T>();

        const EigenIndex bag_number = input_t_->dims()[0];
        const EigenIndex sequence_length = input_t_->dims()[1];
        const EigenIndex output_dim = params_value_t->dims()[1];

        std::unordered_map<IdT, EigenIndex> index_map;
        std::vector< std::pair<IdT, std::vector<EigenIndex> > > index_vec;

        for (EigenIndex i=0; i<bag_number*sequence_length; ++i){
            auto index = indices_d[i];
            if (index_map.find(index) == index_map.end()) {
                index_map[index] = index_vec.size();
                index_vec.push_back({ index,{} });
            }
            index_vec[index_map[index]].second.push_back(i);
        }

        for (EigenIndex i = 0; i < bag_number; ++i) {
            EigenVectorMap<T> params_grads_slice(&params_grad_d[index_vec[i].first * output_dim], output_dim );
      
        for (EigenIndex index : index_vec[i].second) {
            const EigenIndex bag = index / sequence_length;
            const EigenIndex seq = index % sequence_length;
            const ConstEigenVectorMap<T> grads_slice(&output_d[bag*output_dim], output_dim);
            params_grads_slice += grads_slice * weight_value_d[bag*sequence_length + seq];
        }
        if (mode_t == "mean") {
            params_grads_slice /= static_cast<T>(sequence_length);
        }

        }   

        for (EigenIndex i=0; i<bag_number; ++i){

            for (EigenIndex j=0; j<sequence_length; ++j){
                const ConstEigenVectorMap<T> grads_slice( &output_d[i * output_dim ], output_dim );
                const ConstEigenVectorMap<T> params_slice(&params_value_d[indices_d[i*sequence_length+j] * output_dim ], output_dim );
                if (mode_t == "sum"){
                weight_grad_d[i * sequence_length + j]  =   params_slice.dot(grads_slice);
                }else {
                weight_grad_d[i * sequence_length + j] = params_slice.dot(grads_slice) / static_cast<T>(sequence_length);
                }
            
            }
        }
        
    } //apply

    private:
        const framework::ExecutionContext &context_;
        const Tensor *input_t_;
};

template <typename T>
class EmbeddingBagGradKernel : public framework::OpKernel<T> {
    public:
        void Compute(const framework::ExecutionContext &context) const override {
            const auto *input = context.Input<Tensor>("input");
            EmbeddingBagGradCPUFunctor<T> functor(context, input);
            framework::VisitIntDataType(framework::TransToProtoVarType(input->dtype()), functor);

        }
};

} //namespace operators
} //namespace paddle