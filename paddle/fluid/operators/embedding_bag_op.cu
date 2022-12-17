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

# define EIGEN_USE_GPU


#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sort.h>

#include "paddle/fluid/operators/embedding_bag_op.h"

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {

namespace operators {

//global kernel have problem in the judgement of std::string
enum class CalMode {
    ksum, kmean
};

using EigenIndex = Eigen::Index;


template <typename T, typename IdT, int GridDimX >
__global__ void EmbeddingBag(T *output, const T *params, const IdT *input, const T *weight,
                             const EigenIndex output_dim, const EigenIndex sequence_length, 
                             CalMode mode ) {

    EigenIndex feature_idx = threadIdx.x + blockIdx.y * GridDimX;

    while ( feature_idx <= sequence_length) {
        const EigenIndex output_idx = blockIdx.x * output_dim + feature_idx;
        const EigenIndex bag_offset = blockIdx.x * sequence_length;
        T accum = static_cast<T>(0);
        for (EigenIndex idx_offset = bag_offset; idx_offset < bag_offset + sequence_length; ++idx_offset ) {
            accum += params[ input[idx_offset] * output_dim + feature_idx ] * weight[idx_offset];
        }
        if (mode == CalMode::kmean) {
            accum /= static_cast<T>(sequence_length);
        }
        output[output_idx] = accum;
        feature_idx += blockDim.y * GridDimX;

    }
    
    
}

template <typename T>
struct EmbeddingBagCUDAFunctor {
    EmbeddingBagCUDAFunctor(const framework::ExecutionContext &context,
                            const phi::DenseTensor *input_t)
        : context_(context), input_t_(input_t) {}
    
    template <typename IdT>
    void apply() {
        auto *params_t = context_.Input<LoDTensor>("params");
        auto *output_t = context_.Output<LoDTensor>("out");
        auto *weight_t = context_.Input<LoDTensor>("weight");
        std::string mode = context_.Attr<std::string>("mode");
        CalMode mode_enum = CalMode::ksum;
        if (mode == "mean"){
            CalMode mode_enum = CalMode::kmean;
        }

        size_t output_dim = params_t -> dims()[1];
        size_t sequence_length = input_t_ -> dims()[1];
        size_t bag_number = input_t_ -> dims()[0];

        const auto *indices_d = input_t_ -> data<IdT>();
        const auto *params_d = params_t -> data<T>();
        const auto *weight_d = weight_t -> data<T>();
        auto *output_d = output_t -> mutable_data<T>(context_.GetPlace());

        const int kThreadsPerBlock = 32;
        auto stream = context_.cuda_device_context().stream();
        const int blocks_per = Eigen::divup(static_cast<EigenIndex>(output_dim), static_cast<EigenIndex>(kThreadsPerBlock));


        dim3 grids(bag_number, blocks_per);
        
        EmbeddingBag<T,IdT,4><<<grids, kThreadsPerBlock, 0, stream>>>(output_d, params_d, indices_d, weight_d,
                                                           output_dim, sequence_length, mode_enum);
    
    }

    private:
        const framework::ExecutionContext &context_;
        const phi::DenseTensor *input_t_;



};


template <typename T>
class EmbeddingBagCUDAKernel : public framework::OpKernel<T> {
    public:
        void Compute(const framework::ExecutionContext &context) const override {
            const auto *input_t = context.Input<phi::DenseTensor>("input");
            EmbeddingBagCUDAFunctor<T> functor(context, input_t);
            framework::VisitIntDataType(framework::TransToProtoVarType(input_t -> dtype()), functor);


        }
};


template <typename T, typename IdT>
__global__ void EmbeddingBagWeightsGrad(const int output_dim, const IdT *input, const T *params, 
                                        const T *grads, T *weights_grad, CalMode mode) {

    const int bag_idx = blockIdx.y;
    const int sequence_length = gridDim.y;
    const int bag_number = gridDim.x;
    const int sample_idx = blockIdx.x;

    const int paramsIdx = input[ (sample_idx * sequence_length) + bag_idx ] * output_dim;
    const int gradsIdx = sample_idx * output_dim;
    float partialDotProduct = 0.0f;
    for (int i = threadIdx.x; i < output_dim; i ++){
        partialDotProduct += static_cast<float>(params[paramsIdx + i] * grads[gradsIdx + i] );

    }
    if (mode == CalMode::kmean) {
        partialDotProduct /= static_cast<float>(sequence_length);
    }
    
    if (threadIdx.x == 0) {
        weights_grad[ (sample_idx * sequence_length) + bag_idx ] = static_cast<T>(partialDotProduct);
    }
}

template <typename IdT>
__global__ void PrepTempArraysKernel(
    const IdT *indices, IdT *sortedIndices,
    IdT *sortedIndicesCounter, const int indices_size) {
  const int arrayIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (arrayIdx <
      indices_size) {  
    sortedIndices[arrayIdx] = indices[arrayIdx];
    sortedIndicesCounter[arrayIdx] = arrayIdx;
  }
}

//introduce lgd to avoid duplicate reads and writes
template <typename T, typename IdT>
__global__ void EmbeddingBagParamsGrad(const int output_dim, const int sequence_length, const IdT *sortedIndices,  
                                       const IdT * counter, 
                                       const T *weights, const T *grads, T *params_grad, CalMode mode) {

    const int sample_idx = blockIdx.x;
    const int bag_idx = blockIdx.y;
    const int feature_idx = threadIdx.x + bag_idx * blockDim.x;
    
    const int params_idx = __ldg(sortedIndices + sample_idx) ;
    if (sample_idx > 0) {
        const int prev_idx = __ldg(sortedIndices + sample_idx - 1);
        if (prev_idx == params_idx) {
            return;
        }
    }
    int end_idx = sample_idx;
    while (end_idx < gridDim.x - 1) 
    {
        int next_idx = end_idx + 1;
        int next_params_idx = __ldg(sortedIndices + next_idx);
        if (next_params_idx == params_idx) {
            end_idx += 1;
        } else {
            break;
        }
    }

    if (feature_idx < output_dim) { 
        const int outputoffset = (params_idx * output_dim) + feature_idx;
        float accum = 0.0f;

        for ( int i = sample_idx; i <= end_idx; ++i ) {
            int indices_idx = __ldg(counter + i);
            auto weight_slice = weights[indices_idx];
            auto grad_slice =  __ldg( grads + (indices_idx / sequence_length) + feature_idx) ;
            accum += static_cast<float>(weight_slice * grad_slice);
        }
        if (mode == CalMode::kmean) {
            accum /= static_cast<float>(sequence_length);
        }
        params_grad[outputoffset] = static_cast<T>(accum);
    }






}


template <typename T>
struct EmbeddingBagGradCUDAFunctor {
    EmbeddingBagGradCUDAFunctor(const framework::ExecutionContext &context,
                                const phi::DenseTensor *input_t)
        : context_(context), input_t_(input_t) {}
    
    template <typename IdT>
    void apply() {
        
        auto *params_grad_t = context_.Output<phi::DenseTensor>(framework::GradVarName("params"));
        auto *weight_grad_t = context_.Output<phi::DenseTensor>(framework::GradVarName("weight")) ;
        auto *grad_t = context_.Input<phi::DenseTensor>(framework::GradVarName("out"));
        auto *indices_t = context_.Input<phi::DenseTensor>("input");
        auto *params_value_t = context_.Input<phi::DenseTensor>("params");
        auto *weight_value_t = context_.Input<phi::DenseTensor>("weight");

        std::string mode = context_.Attr<std::string>("mode");


        const auto *indices_d = indices_t -> data<IdT>();
        const auto *params_value_d = params_value_t -> data<T>();
        auto *params_grad_d = params_grad_t -> mutable_data<T>(context_.GetPlace());
        const auto *grads_d = grad_t -> data<T>();
        auto *weight_grad_d = weight_grad_t -> mutable_data<T>(context_.GetPlace());
        const auto *weight_value_d = weight_value_t -> data<T>();

        size_t bag_number = indices_t -> dims()[0];
        size_t sequence_length = indices_t -> dims()[1];

        size_t output_dim = params_value_t -> dims()[1];

        auto &dev_ctx = context_.template device_context<phi::GPUContext>();
        const int kThreadsPerBlock = 32;
        dim3 grids(bag_number,sequence_length,1);

        CalMode mode_enum = CalMode::ksum;
        if (mode == "mean"){
            CalMode mode_enum = CalMode::kmean;
        }


        EmbeddingBagWeightsGrad<T, IdT><<<grids, kThreadsPerBlock, 0 ,dev_ctx.stream() >>>(output_dim, 
            indices_d, params_value_d, grads_d, weight_grad_d, mode_enum);
        
        auto &dev_ctx_data = context_.template device_context<phi::GPUContext>();
        phi::DenseTensor sortedIndices =  context_.AllocateTmpTensor<IdT, phi::GPUContext>(
            indices_t->dims(), dev_ctx_data);
        phi::DenseTensor sortedIndicesCounter = context_.AllocateTmpTensor<IdT, phi::GPUContext>(
            indices_t->dims(), dev_ctx_data);
        
        sortedIndices.mutable_data<IdT>(context_.GetPlace());
        sortedIndicesCounter.mutable_data<IdT>(context_.GetPlace());
        
        const int indices_size = indices_t -> dims()[0] * indices_t -> dims()[1];
        const auto params_size = params_value_t -> dims()[0] * params_value_t -> dims()[1];
        const int total_blocks = Eigen::divup(indices_size, kThreadsPerBlock);

        dim3 grids_2(total_blocks,1,1);

        auto *sortedIndices_d = sortedIndices.data<IdT>();
        auto *sortedIndicesCounter_d = sortedIndicesCounter.data<IdT>();
       
        PrepTempArraysKernel<IdT><<<grids_2, kThreadsPerBlock, 0, dev_ctx.stream() >>>(indices_d, 
            sortedIndices_d, sortedIndicesCounter_d, indices_size);

        thrust::device_ptr<IdT> sortedIndicesCounterDevicePtr(sortedIndicesCounter.data<IdT>());
        thrust::device_ptr<IdT> sortedIndicesDevicePtr(sortedIndices.data<IdT>());
        thrust::device_ptr<T> paramsGradDevicePtr(params_grad_d);

        thrust::fill(paramsGradDevicePtr, paramsGradDevicePtr + static_cast<int>(params_size),
                    static_cast<T>(0.0f));
        thrust::sort_by_key(sortedIndicesDevicePtr,
                            sortedIndicesDevicePtr + indices_size,
                            sortedIndicesCounterDevicePtr);
        
        int threadsPerBlock;
        int blocksPerRow;
        if (output_dim <= 1024) {
        blocksPerRow = 1;
        threadsPerBlock = output_dim;
        } else {
        blocksPerRow =
            Eigen::divup(static_cast<int>(output_dim), 1024); //MAX_THREADS_PER_BLOCK
        threadsPerBlock =
            Eigen::divup(static_cast<int>(output_dim), blocksPerRow);
    }

        dim3 grids_3(indices_size, blocksPerRow, 1);
        EmbeddingBagParamsGrad<T, IdT><<<grids_3, kThreadsPerBlock, 0, dev_ctx.stream()>>>(output_dim, sequence_length,
                sortedIndices.data<IdT>(), sortedIndicesCounter.data<IdT>(), weight_value_d, grads_d, params_grad_d, mode_enum );

    }

    private:
        const framework::ExecutionContext &context_;
        const phi::DenseTensor *input_t_;
}; 


template <typename T>
class EmbeddingBagGradCUDAKernel : public framework::OpKernel<T> {
    public:
        void Compute(const framework::ExecutionContext &context) const override {
            const auto *input_t = context.Input<phi::DenseTensor>("input");
            EmbeddingBagGradCUDAFunctor<T> functor(context, input_t);
            framework::VisitIntDataType(framework::TransToProtoVarType(input_t -> dtype()), functor);
        }
};

} //namespace operators


} //namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    embedding_bag, ops::EmbeddingBagCUDAKernel<float>,
    ops::EmbeddingBagCUDAKernel<double>,
    ops::EmbeddingBagCUDAKernel<int8_t>,
    ops::EmbeddingBagCUDAKernel<int16_t>);

REGISTER_OP_CUDA_KERNEL(
    embedding_bag_grad, ops::EmbeddingBagGradCUDAKernel<float>,
    ops::EmbeddingBagGradCUDAKernel<double>
)