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

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"

#include "paddle/common/ddim.h"
#include "paddle/common/macros.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/ir_printer.h"

REGISTER_FILE_SYMBOLS(kernel_dialect);

namespace paddle::dialect {

void PrintKernelType(pir::Type type, std::ostream &os) {
  if (type.isa<AllocatedDenseTensorType>()) {
    AllocatedDenseTensorType tensor_type =
        type.dyn_cast<AllocatedDenseTensorType>();

    os << phi::AllocationTypeStr(tensor_type.place().GetType()) << "_";
    os << "tensor<";
    for (auto d : common::vectorize(tensor_type.dims())) {
      os << d;
      os << "x";
    }
    tensor_type.dtype().Print(os);
    os << ">";
  } else if (type.isa<AllocatedSelectedRowsType>()) {
    AllocatedSelectedRowsType tensor_type =
        type.dyn_cast<AllocatedSelectedRowsType>();

    os << phi::AllocationTypeStr(tensor_type.place().GetType()) << "_";
    os << "tensor<";
    for (auto d : common::vectorize(tensor_type.dims())) {
      os << d;
      os << "x";
    }
    tensor_type.dtype().Print(os);
    os << ">";
  } else if (type.isa<AllocatedDenseTensorArrayType>()) {
    AllocatedDenseTensorArrayType tensor_array_type =
        type.dyn_cast<AllocatedDenseTensorArrayType>();

    os << phi::AllocationTypeStr(tensor_array_type.place().GetType()) << "_";
    os << "tensor_array<";
    tensor_array_type.dtype().Print(os);
    os << ">";
  } else if (type.isa<AllocatedSparseCooTensorType>()) {
    AllocatedSparseCooTensorType sparse_coo_tensor_type =
        type.dyn_cast<AllocatedSparseCooTensorType>();

    os << phi::AllocationTypeStr(sparse_coo_tensor_type.place().GetType())
       << "_";
    os << "sparse_coo_tensor_type<";
    sparse_coo_tensor_type.dtype().Print(os);
    os << ">";
  } else if (type.isa<AllocatedSparseCsrTensorType>()) {
    AllocatedSparseCsrTensorType sparse_csr_tensor_type =
        type.dyn_cast<AllocatedSparseCsrTensorType>();

    os << phi::AllocationTypeStr(sparse_csr_tensor_type.place().GetType())
       << "_";
    os << "sparse_csr_tensor_type<";
    sparse_csr_tensor_type.dtype().Print(os);
    os << ">";
  }
}

void PrintKernelAttribute(pir::Attribute attr, std::ostream &os) {
  phi::KernelKey kernel = attr.dyn_cast<KernelAttribute>().data();

  os << "<backend:" << kernel.backend() << "|layout:" << kernel.layout()
     << "|dtype:" << kernel.dtype() << ">";
}

KernelDialect::KernelDialect(pir::IrContext *context)
    : pir::Dialect(name(), context, pir::TypeId::get<KernelDialect>()) {
  initialize();
}

void KernelDialect::initialize() {
  RegisterTypes<paddle::dialect::AllocatedDenseTensorType,
                paddle::dialect::AllocatedSelectedRowsType,
                paddle::dialect::AllocatedSparseCooTensorType,
                paddle::dialect::AllocatedSparseCsrTensorType,
                paddle::dialect::AllocatedDenseTensorArrayType>();
  RegisterOps<dialect::PhiKernelOp, dialect::LegacyKernelOp>();
  RegisterAttributes<paddle::dialect::KernelAttribute>();
}

void KernelDialect::PrintType(pir::Type type, std::ostream &os) const {
  PrintKernelType(type, os);
}

void KernelDialect::PrintAttribute(pir::Attribute attr,
                                   std::ostream &os) const {
  PrintKernelAttribute(attr, os);
}

pir::OpPrintFn KernelDialect::PrintOperation(const pir::Operation &op) const {
  if (op.dyn_cast<PhiKernelOp>() || op.dyn_cast<LegacyKernelOp>()) {
    return [](const pir::Operation &op, pir::IrPrinter &printer) {
      auto &os = printer.os;
      printer.PrintOpResult(op);
      os << " =";
      if (auto phi_kernel_op = op.dyn_cast<PhiKernelOp>()) {
        std::string kernel_name = phi_kernel_op.kernel_name();
        if (op.attributes().count("is_inplace") != 0 &&
            op.attributes()
                .at("is_inplace")
                .dyn_cast<pir::BoolAttribute>()
                .data()) {
          kernel_name = kernel_name + "_";
        }
        os << " \"" << kernel_name << "(phi_kernel)\"";
      } else {
        auto legacy_kernel_op = op.dyn_cast<LegacyKernelOp>();
        std::string kernel_name = legacy_kernel_op.kernel_name();
        if (op.attributes().count("is_inplace") != 0 &&
            op.attributes()
                .at("is_inplace")
                .dyn_cast<pir::BoolAttribute>()
                .data()) {
          kernel_name = kernel_name + "_";
        }
        os << " \"" << kernel_name << "(legacy_kernel)\"";
      }
      printer.PrintOpOperands(op);
      printer.PrintAttributeMap(op);
      os << " :";
      printer.PrintOperandsType(op);
      os << " -> ";
      printer.PrintOpReturnType(op);
    };
  }
  return nullptr;
}

CustomKernelDialect::CustomKernelDialect(pir::IrContext *context)
    : pir::Dialect(name(), context, pir::TypeId::get<CustomKernelDialect>()) {
  initialize();
}

void CustomKernelDialect::initialize() {
  RegisterOps<dialect::CustomKernelOp>();
}

void CustomKernelDialect::PrintType(pir::Type type, std::ostream &os) const {
  PrintKernelType(type, os);
}

void CustomKernelDialect::PrintAttribute(pir::Attribute attr,
                                         std::ostream &os) const {
  PrintKernelAttribute(attr, os);
}

pir::OpPrintFn CustomKernelDialect::PrintOperation(
    const pir::Operation &op) const {
  return [](const pir::Operation &op, pir::IrPrinter &printer) {
    auto &os = printer.os;
    printer.PrintOpResult(op);
    os << " =";
    auto custom_kernel_op = op.dyn_cast<CustomKernelOp>();
    std::string kernel_name = custom_kernel_op.kernel_name();
    if (op.attributes().count("is_inplace") != 0 &&
        op.attributes()
            .at("is_inplace")
            .dyn_cast<pir::BoolAttribute>()
            .data()) {
      kernel_name = kernel_name + "_";
    }
    os << " \"" << kernel_name << "(custom_kernel)\"";
    printer.PrintOpOperands(op);
    printer.PrintAttributeMap(op);
    os << " :";
    printer.PrintOperandsType(op);
    os << " -> ";
    printer.PrintOpReturnType(op);
  };
}
#ifdef PADDLE_WITH_DNNL
OneDNNKernelDialect::OneDNNKernelDialect(pir::IrContext *context)
    : pir::Dialect(name(), context, pir::TypeId::get<OneDNNKernelDialect>()) {
  initialize();
}

void OneDNNKernelDialect::initialize() {
  RegisterOps<dialect::OneDNNPhiKernelOp,
              dialect::OneDNNMixedPhiKernelOp,
              dialect::OneDNNLegacyKernelOp>();
}

void OneDNNKernelDialect::PrintType(pir::Type type, std::ostream &os) const {
  PrintKernelType(type, os);
}

void OneDNNKernelDialect::PrintAttribute(pir::Attribute attr,
                                         std::ostream &os) const {
  PrintKernelAttribute(attr, os);
}

pir::OpPrintFn OneDNNKernelDialect::PrintOperation(
    const pir::Operation &op) const {
  if (op.dyn_cast<PhiKernelOp>() || op.dyn_cast<LegacyKernelOp>()) {
    return [](const pir::Operation &op, pir::IrPrinter &printer) {
      auto &os = printer.os;
      printer.PrintOpResult(op);
      os << " =";
      if (auto phi_kernel_op = op.dyn_cast<PhiKernelOp>()) {
        std::string kernel_name = phi_kernel_op.kernel_name();
        if (op.attributes().count("is_inplace") != 0 &&
            op.attributes()
                .at("is_inplace")
                .dyn_cast<pir::BoolAttribute>()
                .data()) {
          kernel_name = kernel_name + "_";
        }
        os << " \"" << kernel_name << "(phi_kernel)\"";
      } else {
        auto legacy_kernel_op = op.dyn_cast<LegacyKernelOp>();
        std::string kernel_name = legacy_kernel_op.kernel_name();
        if (op.attributes().count("is_inplace") != 0 &&
            op.attributes()
                .at("is_inplace")
                .dyn_cast<pir::BoolAttribute>()
                .data()) {
          kernel_name = kernel_name + "_";
        }
        os << " \"" << kernel_name << "(legacy_kernel)\"";
      }
      printer.PrintOpOperands(op);
      printer.PrintAttributeMap(op);
      os << " :";
      printer.PrintOperandsType(op);
      os << " -> ";
      printer.PrintOpReturnType(op);
    };
  }
  return nullptr;
}
#endif

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::KernelDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::CustomKernelDialect)
#ifdef PADDLE_WITH_DNNL
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::OneDNNKernelDialect)
#endif
