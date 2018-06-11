/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "Function.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/SparseMatrix.h"

namespace paddle {
/// CPU, dense matrix (+)= dense matrix * dense matrix
template <DeviceType DType>
void MulOp(CpuMatrix& out,
           const CpuMatrix& a,
           const CpuMatrix& b,
           real scaleAB,
           real scaleT,
           bool aTrans,
           bool bTrans);

/// CPU, dense matrix (+)= sparse matrix * dense matrix
template <DeviceType DType>
void MulOp(CpuMatrix& out,
           const CpuSparseMatrix& a,
           const CpuMatrix& b,
           real scaleAB,
           real scaleT,
           bool aTrans,
           bool bTrans);

/// CPU, dense matrix (+)= dense matrix * sparse matrix
template <DeviceType DType>
void MulOp(CpuMatrix& out,
           const CpuMatrix& a,
           const CpuSparseMatrix& b,
           real scaleAB,
           real scaleT,
           bool aTrans,
           bool bTrans);

/// CPU, sparse matrix (+)= dense matrix * dense matrix
template <DeviceType DType>
void MulOp(CpuSparseMatrix& out,
           const CpuMatrix& a,
           const CpuMatrix& b,
           real scaleAB,
           real scaleT,
           bool aTrans,
           bool bTrans);

/// GPU, dense matrix (+)= dense matrix * dense matrix
template <DeviceType DType>
void MulOp(GpuMatrix& out,
           const GpuMatrix& a,
           const GpuMatrix& b,
           real scaleAB,
           real scaleT,
           bool aTrans,
           bool bTrans);

/// GPU, dense matrix (+)= sparse matrix * dense matrix
template <DeviceType DType>
void MulOp(GpuMatrix& out,
           const GpuSparseMatrix& a,
           const GpuMatrix& b,
           real scaleAB,
           real scaleT,
           bool aTrans,
           bool bTrans);

/// GPU, dense matrix (+)= dense matrix * sparse matrix
template <DeviceType DType>
void MulOp(GpuMatrix& out,
           const GpuMatrix& a,
           const GpuSparseMatrix& b,
           real scaleAB,
           real scaleT,
           bool aTrans,
           bool bTrans);

/// GPU, sparse matrix (+)= dense matrix * dense matrix
template <DeviceType DType>
void MulOp(GpuSparseMatrix& out,
           const GpuMatrix& a,
           const GpuMatrix& b,
           real scaleAB,
           real scaleT,
           bool aTrans,
           bool bTrans);

}  // namespace paddle
