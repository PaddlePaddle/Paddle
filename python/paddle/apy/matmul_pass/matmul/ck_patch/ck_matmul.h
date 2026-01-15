// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h> 

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/amd_buffer_addressing.hpp"

#include "ck_patch/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_dl_variadic.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck_patch/tensor_operation/gpu/element/variadic_element_wise_operation.hpp"
#include "ck_patch/batched_matrix_coord.h"
#include "ck_patch/all_tuning_configs.h"
#include "ck_patch/utility/unroll.hpp"
#include "params.h"

namespace ap {

using MatrixCoord = ck::BatchedMatrixCoord;
using bfloat16 = hip_bfloat16;

template <typename T, int VecSize>
using VectorType = ck::vector_type<typename ck::CKDataType<T>::Type, VecSize>;

template <int NUnroll>
using unroll = ck::unroll<NUnroll>;

template <typename T, int VecSize>
__device__ __forceinline__ auto load_vector(const T* ptr, int64_t offset, bool valid, int64_t size) {
    using CKType = typename ck::CKDataType<T>::Type;
    return ck::amd_buffer_load_invalid_element_return_zero<CKType, VecSize>(reinterpret_cast<const CKType*>(ptr), offset, valid, size);
}

template <typename T, int VecSize, int I>
__device__ __forceinline__ constexpr const auto& 
extract_scalar(const ck::vector_type<T, VecSize>& vec, ck::Number<I> i) {
    return vec.template AsType<T>()(i);
}

template <typename T, int VecSize, int I>
__device__ __forceinline__ constexpr auto& 
extract_scalar(ck::vector_type<T, VecSize>& vec, ck::Number<I> i) {
    return vec.template AsType<T>()(i);
}

}

namespace ck {
template <typename Len, int Align, int dim0, int dim1>
struct AlignBlockTransferTensor {
private:
    using D0 = Number<dim0>;
    using D1 = Number<dim1>;
    static constexpr int v0 = Len::At(dim0);
    static constexpr int v1 = Len::At(dim1);

public:
    using type = std::conditional_t<
        (v1 >= Align), 
        decltype(Len::Modify(D0{}, Number<1>{}).Modify(D1{}, Number<Align>{})),
        decltype(Len::Modify(D0{}, Number<std::min(Align / v1, v0)>{}))
    >;
};
}

namespace ap{

template <typename ElementT,
          typename ElementComputeT,
          template <typename T>
          class VariadicFunctor,
          int AlignA,
          int AlignB,
          int ConfigId = DefaultConfig::kConfigId>
void MatmulAddVariadic(
        const GemmEpilogueParams &params,
        const typename VariadicFunctor<ElementComputeT>::Arguments &variadic_args
    ) {

    using Row = ck::tensor_layout::gemm::RowMajor;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ALayout  = Row;
    using BLayout  = Row;
    using DsLayout = ck::Tuple<>;
    using ELayout  = Row; 

    using ADataType = typename ck::CKDataType<ElementT>::Type;
    using BDataType = typename ck::CKDataType<ElementT>::Type;
    using AccDataType = typename ck::CKDataType<ElementComputeT>::Type;
    using DsDataType = ck::Tuple<>;
    using EDataType = typename ck::CKDataType<ElementT>::Type;

    using AOrder = S<1,2,0,3>;
    using BOrder = S<0,3,1,2>;

    using Config = GemmTuningConfigs<ElementT, ConfigId>;
    using AlignedABlockTransferSrcVectorTensorLengths = 
        typename ck::AlignBlockTransferTensor<typename Config::ABlockTransferSrcVectorTensorLengths, AlignA, 0, 3>::type;
    using AlignedBBlockTransferSrcVectorTensorLengths = 
        typename ck::AlignBlockTransferTensor<typename Config::BBlockTransferSrcVectorTensorLengths, AlignB, 1, 2>::type;

    static constexpr int ALignedCThreadTransferDstScalarPerVector = std::min(Config::CThreadTransferDstScalarPerVector, AlignB);
    static constexpr int VecSize = ALignedCThreadTransferDstScalarPerVector;

    using AElementOp   = PassThrough;
    using BElementOp   = PassThrough;
    using CDEElementOp = ck::tensor_operation::element_wise::VariadicElementwiseOp<VariadicFunctor, AccDataType, VecSize>;

    static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNPadding;

    using DeviceOpInstance = ck::tensor_operation::device::DeviceBatchedGemmMultipleD_Dl_WithVariadic<
        ALayout, BLayout, DsLayout, ELayout, 
        ADataType, BDataType, AccDataType, DsDataType, EDataType,  
        AElementOp, BElementOp, CDEElementOp, 
        GemmDefault,   
        Config::kBlockSize,
        Config::kMPerBlock,
        Config::kNPerBlock,
        Config::kK0PerBlock,
        Config::kK1,
        Config::kM1PerThreadM111,
        Config::kN1PerThreadN111,
        1,
        typename Config::MThreadCluster,
        typename Config::NThreadCluster,
        typename Config::ABlockTransferThreadSliceLengths,
        typename Config::ABlockTransferThreadClusterLengths,
        AOrder,
        AOrder,
        AlignedABlockTransferSrcVectorTensorLengths,
        AOrder,
        typename Config::ABlockTransferDstVectorTensorLengths,
        typename Config::BBlockTransferThreadSliceLengths,
        typename Config::BBlockTransferThreadClusterLengths,
        BOrder,
        BOrder,
        AlignedBBlockTransferSrcVectorTensorLengths,
        BOrder,
        typename Config::BBlockTransferDstVectorTensorLengths,
        S<0, 1, 2, 3, 4, 5>,
        5,
        ALignedCThreadTransferDstScalarPerVector>;

        auto a_element_op   = AElementOp{};
        auto b_element_op   = BElementOp{};
        auto cde_element_op = CDEElementOp(variadic_args);

        auto device_op = DeviceOpInstance{};
        auto invoker   = device_op.MakeInvoker();
        auto argument =
            device_op.MakeArgument(params.input,
                                params.weight,
                                {}, // TODO check bias
                                params.output,
                                params.m,
                                params.n,
                                params.k,
                                params.batch_count,
                                params.shape_args.lda,
                                params.shape_args.ldb,
                                {},
                                params.shape_args.ldd,
                                params.shape_args.batch_stride_A,
                                params.shape_args.batch_stride_B,
                                {},
                                params.shape_args.batch_stride_D,
                                a_element_op,
                                b_element_op,
                                cde_element_op);

    if(!device_op.IsSupportedArgument(argument)){
        std::cout << "wrong! this device_op instance does not support this problem" << std::endl;
        exit(-1);
    }

    hipStream_t* stream_ptr = reinterpret_cast<hipStream_t*>(params.stream_ptr);
    invoker.Run(argument, StreamConfig{*stream_ptr, false});
}

}	// namespace ap
