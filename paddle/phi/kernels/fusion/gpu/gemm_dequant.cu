// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/fusion/gpu/gemm_dequant.h"
#include "cutlass/bfloat16.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/half.h"

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T>
struct CutlassDtypeTraits {
  using DataT = T;
};

template <>
struct CutlassDtypeTraits<__nv_bfloat16> {
  using DataT = cutlass::bfloat16_t;
};

template <>
struct CutlassDtypeTraits<phi::dtype::bfloat16> {
  using DataT = cutlass::bfloat16_t;
};

template <>
struct CutlassDtypeTraits<half> {
  using DataT = cutlass::half_t;
};

template <>
struct CutlassDtypeTraits<phi::dtype::float16> {
  using DataT = cutlass::half_t;
};

template <typename T>
void RunGemmDequant(const int8_t* a,
                    const int8_t* b,  // Transposed
                    const float* dequant_scale,
                    T* c,
                    int m,
                    int k,
                    int n,
                    cudaStream_t stream) {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = typename CutlassDtypeTraits<T>::DataT;
  using ElementCompute = int32_t;
  using ElementD = ElementC;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;

  static int const kStages = 5;

  /// Linear scaling operator
  using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementCompute,
      ElementCompute>;

  using GemmDequantT = cutlass::GemmDequant<ElementA,
                                            LayoutA,
                                            ElementB,
                                            LayoutB,
                                            ElementC,
                                            ElementCompute,
                                            OperatorClass,
                                            ArchTag,
                                            ThreadblockShape,
                                            WarpShape,
                                            InstructionShape,
                                            EpilogueFunctorOp,
                                            kStages>;

  using LayoutC = typename GemmDequantT::LayoutC;

  int64_t lda = LayoutA::packed({m, k}).stride(0);
  int64_t ldb = LayoutB::packed({k, n}).stride(0);
  int64_t ldc = LayoutC::packed({m, n}).stride(0);

  cutlass::gemm::GemmCoord problem_size(m, n, k);

  typename CutlassDtypeTraits<T>::DataT* c_tmp = nullptr;
  typename CutlassDtypeTraits<T>::DataT* d =
      reinterpret_cast<typename CutlassDtypeTraits<T>::DataT*>(c);

  typename GemmDequantT::TensorRefA ref_a(const_cast<int8_t*>(a), lda);
  typename GemmDequantT::TensorRefB ref_b(const_cast<int8_t*>(b), ldb);
  typename GemmDequantT::TensorRefC ref_c(c_tmp, ldc);
  typename GemmDequantT::TensorRefC ref_d(d, ldc);
  typename GemmDequantT::TensorRefScale ref_scale(
      const_cast<float*>(dequant_scale), 0);

  typename GemmDequantT::Arguments args(
      problem_size,
      ref_a,
      ref_b,
      ref_c,
      ref_d,
      ref_scale,
      {ElementCompute(1.0f), ElementCompute(0.0f)});

  GemmDequantT gemm;
  // Initialize
  auto status = gemm.initialize(args);
  // PADDLE_ENFORCE_EQ(status, cutlass::Status::kSuccess,
  // paddle::platform::errors::Fatal("cutlass GemmDequant initialize error"));

  // Run
  status = gemm(stream);
  // PADDLE_ENFORCE_EQ(status, cutlass::Status::kSuccess,
  // paddle::platform::errors::Fatal("cutlass GemmDequant runtime error"));
}

template <typename T, typename Context>
void GemmDequantKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dequant_out_scales,
                       bool bfloat16_out,
                       DenseTensor* out) {
  std::vector<int64_t> x_dims = common::vectorize(x.dims());
  std::vector<int64_t> y_dims = common::vectorize(y.dims());
  int64_t m = x_dims[x_dims.size() - 2];
  int64_t k = x_dims[x_dims.size() - 1];
  int64_t n = y_dims[y_dims.size() - 2];
  out->Resize({{m, n}});
  if (bfloat16_out) {
    dev_ctx.template Alloc<phi::dtype::bfloat16>(out);
    RunGemmDequant<phi::dtype::bfloat16>(x.data<int8_t>(),
                                         y.data<int8_t>(),
                                         dequant_out_scales.data<float>(),
                                         out->data<phi::dtype::bfloat16>(),
                                         m,
                                         k,
                                         n,
                                         dev_ctx.stream());
  } else {
    dev_ctx.template Alloc<phi::dtype::float16>(out);
    RunGemmDequant<phi::dtype::float16>(x.data<int8_t>(),
                                        y.data<int8_t>(),
                                        dequant_out_scales.data<float>(),
                                        out->data<phi::dtype::float16>(),
                                        m,
                                        k,
                                        n,
                                        dev_ctx.stream());
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(
    gemm_dequant, GPU, ALL_LAYOUT, phi::fusion::GemmDequantKernel, int8_t) {}
