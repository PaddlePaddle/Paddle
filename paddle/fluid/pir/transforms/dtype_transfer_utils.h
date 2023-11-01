// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/kernel_factory.h"
#include "paddle/pir/core/op_result.h"
#include "paddle/pir/core/operation.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

namespace pir {
phi::Kernel* GetKernel(pir::Operation* op, const phi::KernelKey& kernel_key);

bool NeedTransformDataType(const phi::DataType& l, const phi::DataType& r);

const phi::DataType GetKernelTypeforVar(
    pir::Operation* op,
    const std::string& var_name,
    const phi::DataType& tensor_dtype,
    const phi::KernelKey* expected_kernel_key);
pir::OpResult AddDtypeTransferOp(pir::Value in,
                                 pir::Block* block,
                                 const phi::KernelKey& kernel_key,
                                 const phi::Place& out_place,
                                 const phi::DataType& src_dtype,
                                 const phi::DataType& dst_dtype);

pir::Type BuildDtypeTransferOutputType(pir::Type type,
                                       const phi::Place& place,
                                       phi::DataType data_dtype,
                                       pir::IrContext* ctx);

}  // namespace pir
