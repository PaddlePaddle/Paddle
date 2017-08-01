/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/eigen.h"
#include "paddle/framework/net.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using OpKernel = framework::OpKernel;
using InferShapeContext = framework::InferShapeContext;
using ExecutionContext = framework::ExecutionContext;
using Variable = framework::Variable;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;
template <typename T,
          size_t D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using Tensor = framework::Tensor;
using OperatorWithKernel = framework::OperatorWithKernel;
using OpProtoAndCheckerMaker = framework::OpProtoAndCheckerMaker;
using OpProto = framework::OpProto;
using OpAttrChecker = framework::OpAttrChecker;
using CPUPlace = platform::CPUPlace;
using GPUPlace = platform::GPUPlace;
using NetOp = framework::NetOp;
using OpRegistry = framework::OpRegistry;
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
