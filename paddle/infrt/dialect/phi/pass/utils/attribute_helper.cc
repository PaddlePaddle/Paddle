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

#include "paddle/infrt/dialect/phi/pass/utils/attribute_helper.h"
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"

namespace infrt {
namespace dialect {

template <typename T>
T GetAttrFromWeightOp(::mlir::Operation* op, llvm::StringRef name) {
  auto attr = op->getAttr(name);
  if (!attr) {
    LOG(FATAL) << "The operation " << op->getName().getStringRef().str()
               << " has not attr " << name.str();
  }
  T attr_with_type = attr.dyn_cast<T>();
  if (!attr_with_type) {
    LOG(FATAL) << "The attribute " << name.str() << " of operation "
               << op->getName().getStringRef().str()
               << "has incompatible type.";
  }
  return attr_with_type;
}

::phi::DenseTensor CreateDenseTensorFromWeightOp(::phi::Allocator* allocator,
                                                 ::mlir::Operation* op) {
  const auto& dims_attr = GetAttrFromWeightOp<::mlir::ArrayAttr>(op, "dims");
  const auto& layout_attr =
      GetAttrFromWeightOp<::infrt::LayoutAttr>(op, "layout");
  const auto& lod_attr = GetAttrFromWeightOp<::mlir::ArrayAttr>(op, "lod");
  const auto& values_attr =
      GetAttrFromWeightOp<::mlir::ArrayAttr>(op, "values");

  CHECK(layout_attr.getLayout() == ::infrt::LayoutType::NCHW);
  CHECK_EQ(lod_attr.size(), 1U);
  CHECK_EQ(lod_attr[0].cast<::mlir::IntegerAttr>().getInt(), 0);

  std::vector<int> dims;
  for (size_t i = 0; i < dims_attr.size(); ++i) {
    dims.push_back(dims_attr[i].cast<::mlir::IntegerAttr>().getInt());
  }

  CHECK_GT(values_attr.size(), 0U);
  CHECK(values_attr[0].getType().isF32());

  ::phi::DenseTensorMeta meta{::phi::DataType::FLOAT32,
                              ::phi::make_ddim(std::move(dims))};
  ::phi::DenseTensor tensor{allocator, std::move(meta)};
  auto* data = tensor.data<float>();

  for (size_t i = 0; i < values_attr.size(); ++i) {
    data[i] = values_attr[i].cast<::mlir::FloatAttr>().getValueAsDouble();
  }

  return tensor;
}

::infrt::phi::CreateCPUContextOp CreateCPUContextOp(::mlir::OpBuilder builder,
                                                    ::mlir::Location loc) {
  return builder.create<infrt::phi::CreateCPUContextOp>(
      loc,
      infrt::phi::ContextType::get(builder.getContext(),
                                   infrt::TargetType::CPU));
}

::infrt::phi::CreateHostInitedDenseTensorOp CreateWeightOpFromDenseTensor(
    ::mlir::OpBuilder builder,
    ::mlir::Location loc,
    ::mlir::Value context,
    const ::phi::DenseTensor& src) {
  auto output_type =
      infrt::DenseTensorType::get(builder.getContext(),
                                  ::infrt::TargetType::CPU,
                                  ::infrt::PrecisionType::FLOAT32,
                                  ::infrt::LayoutType::NCHW);
  auto dims = builder.getI64ArrayAttr(
      ::llvm::ArrayRef<int64_t>(src.dims().Get(), src.dims().size()));
  auto layout =
      ::infrt::LayoutAttr::get(builder.getContext(), ::infrt::LayoutType::NCHW);
  auto lod = builder.getI64ArrayAttr({0});
  auto data = builder.getF32ArrayAttr(
      {src.data<float>(), static_cast<size_t>(src.numel())});
  return builder.create<::infrt::phi::CreateHostInitedDenseTensorOp>(
      loc, output_type, context, dims, layout, lod, data);
}

}  // namespace dialect
}  // namespace infrt
