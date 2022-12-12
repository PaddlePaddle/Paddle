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

#include "paddle/phi/kernels/embedding_bag_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
# define EIGEN_USE_GPU
namespace phi {

using EigenIndex = Eigen::Index; 

template <typename T, typename IdT, int kThreadsPerBlock >
__global__ void EmbeddingBag(T *output, const T *params, const IdT *input, const T *weight,
                             const int output_dim, const int sequence_length, 
                             CalMode mode ) {

    int feature_idx = threadIdx.x + blockIdx.y * kThreadsPerBlock;

    while (feature_idx <= sequence_length) {
        const int output_idx = blockIdx.x * output_dim + feature_idx;
        const int bag_offset = blockIdx.x * sequence_length;
        T accum = static_cast<T>(0);
        for (int idx_offset = bag_offset; idx_offset < bag_offset + sequence_length; ++idx_offset ) {
            accum += params[ input[idx_offset] * output_dim + feature_idx ] * weight[idx_offset];
        }
        if (mode == CalMode::kmean) {
            accum /= static_cast<T>(sequence_length);
        }
        output[output_idx] = accum;
        feature_idx += blockDim.y * kThreadsPerBlock;
    }
    
    
}

template <typename T, typename Context>
struct EmbeddingBagCUDAFunctor {
    EmbeddingBagCUDAFunctor(const Context& dev_ctx,
                            const DenseTensor& input,
                            const DenseTensor& params,
                            const DenseTensor& weight,
                            const std::string& mode,
                            DenseTensor* out) 
        : dev_ctx_(dev_ctx),
          input_(input),
          params_(params),
          weight_(weight),
          mode_(mode),
          out_(out) {}
    
    template <typename IdT>
    void apply() {
        dev_ctx_.template Alloc<T>(out_);

        static constexpr int kThreadsPerBlock = 32;

        int output_dim = params_.dims()[1];
        int sequence_length = input_.dims()[1];
        const int bag_number = input_.dims()[0];

        const T *params_d = params_.data<T>();
        const IdT *indices_d = input_.data<IdT>();
        const T *weight_d = weight_.data<T>();
        T *output_d = out_->data<T>();

        const int blocks_per = Eigen::divup(static_cast<EigenIndex>(output_dim), static_cast<EigenIndex>(32));
        dim3 grids(bag_number, blocks_per);
        auto stream = dev_ctx_.stream();

        CalMode mode_enum = CalMode::ksum;
        if (mode_ == "mean"){
            CalMode mode_enum = CalMode::kmean;
        }

        EmbeddingBag<T, IdT, kThreadsPerBlock><<<grids, kThreadsPerBlock, 0, stream>>>(
            output_d, params_d, indices_d, weight_d, output_dim,
            sequence_length, mode_enum
        );
    }

    private:
        const phi::GPUContext &dev_ctx_;
        const DenseTensor &input_;
        const DenseTensor &params_;
        const DenseTensor &weight_;
        const std::string mode_;
        DenseTensor *out_;


}; // struct

template <typename T, typename Context>
void EmbeddingBagCUDAKernel(const Context &ctx,
                        const DenseTensor &input,
                        const DenseTensor &params,
                        const DenseTensor &weight,
                        const std::string &mode,
                        DenseTensor *out) {
    
    EmbeddingBagCUDAFunctor<T, Context> functor(
        ctx, input, params, weight, mode, out);
    
    if (input.dtype() == phi::DataType::INT32)
    { 
        functor.template apply<int>();
    } else if (input.dtype() == phi::DataType::INT64)
    {
        functor.template apply<int64_t>();
    } else if (input.dtype() == phi::DataType::INT16) {
        functor.template apply<int16_t>();
    } else {
        PADDLE_THROW(phi::errors::Unimplemented( "embebddingbag input only support int32 and int64" ) );
    }


    }

} // namespace phi

PD_REGISTER_KERNEL(embedding_bag,
                   GPU,
                   ALL_LAYOUT,
                   phi::EmbeddingBagCUDAKernel,
                   float,
                   double) {}