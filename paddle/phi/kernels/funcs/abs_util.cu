// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/abs_util.h"
// #include "paddle/phi/kernels/funcs/complex_functors.h"

namespace phi {
namespace funcs {

// template <typename T, typename Func>
// void abs( const DenseTensor& x, DenseTensor* out, Func func )

// template void abs<float, UnaryFunctor()>( const DenseTensor& x, DenseTensor*
// out, UnaryFunctor() func ); template void abs<double, UnaryFunctor()>( const
// DenseTensor& x, DenseTensor* out, UnaryFunctor() func ); template void
// abs<int>( const DenseTensor& x, DenseTensor* out ); template void
// abs<int64_t>( const DenseTensor& x, DenseTensor* out ); template void
// abs<phi::dtype::float16>( const DenseTensor& x, DenseTensor* out ); template
// void abs<phi::dtype::bfloat16>( const DenseTensor& x, DenseTensor* out );
// template void abs<phi::dtype::complex<float>>( const DenseTensor& x,
// DenseTensor* out ); template void abs<phi::dtype::complex<double>>( const
// DenseTensor& x, DenseTensor* out );

// template __global__ void abs_kernel<float>( const float* x, const int num,
// float* out); template __global__ void abs_kernel<double>( const double* x,
// const int num, double* out); template __global__ void abs_kernel<int>( const
// int* x, const int num, int* out); template __global__ void
// abs_kernel<int64_t>(const int64_t* x, const int num, int64_t* out ); template
// __global__ void abs_kernel<phi::dtype::float16>( const phi::dtype::float16*
// x, const int num, phi::dtype::float16* out); template __global__ void
// abs_kernel<phi::dtype::bfloat16>(const phi::dtype::bfloat16* x, const int
// num, phi::dtype::bfloat16* out); template __global__ void
// abs_kernel<phi::dtype::complex<float>>(const phi::dtype::complex<float>* x,
// const int num, phi::dtype::complex<float>* out); template __global__ void
// abs_kernel<phi::dtype::complex<double>>(const phi::dtype::complex<double>* x,
// const int num, phi::dtype::complex<double>* out );
}  // namespace funcs
}  // namespace phi
