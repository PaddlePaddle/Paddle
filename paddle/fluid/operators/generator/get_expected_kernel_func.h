/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/kernel_factory.h"

namespace paddle {
namespace operators {

phi::KernelKey GetReduceExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr);

phi::KernelKey GetReduceGradExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr);

phi::KernelKey GetAssignExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr);

phi::KernelKey GetSgdExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr);

phi::KernelKey GetSoftmaxExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr);

phi::KernelKey GetSoftmaxGradExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr);

phi::KernelKey GetUpdateLossScalingExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr);

phi::KernelKey GetMatrixNmsExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr);

phi::KernelKey GetUniqueExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr);

phi::KernelKey GetYoloLossExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr);

}  // namespace operators
}  // namespace paddle
