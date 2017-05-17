#pragma once
#include "FunctionList.h"
#include "paddle/topology/Attribute.h"
#include "paddle/topology/Function.h"
#include "paddle/topology/Tensor.h"
#include "paddle/topology/meta/TypeDefs.h"

namespace paddle {
namespace function {
using paddle::topology::meta::Set;
using paddle::topology::meta::Map;

namespace details {
typedef std::function<Error(const BufferArgs& ins,
                            const BufferArgs& outs,
                            const topology::AttributeMap& attrs)>
    FunctionWithAttrs;
}

Function createFunction(const topology::Function& conf);

template <typename AttributeType>
class MetaInfoRegister {
public:
  MetaInfoRegister(const char* name)
      : func_(paddle::topology::meta::FunctionMeta::registerFuncMeta(name)) {
    AttributeType::registerFunctionAttribute(func_);
  }

protected:
  typedef topology::meta::FunctionMeta::TensorShapeInferer TensorShapeInferer;
  typedef topology::meta::FunctionTensorType FunctionTensorType;
  typedef topology::meta::FunctionMeta::TensorShapeInfererWithAttrs
      TensorShapeInfererWithAttrs;

  static constexpr FunctionTensorType INPUT =
      topology::meta::FunctionTensorType::INPUT;
  static constexpr FunctionTensorType OUTPUT =
      topology::meta::FunctionTensorType::OUTPUT;

  void setDescription(const std::string& description) {
    func_->metaAttributes_.set("description", description).check();
  }

  void setShapeInferer(TensorShapeInferer inferer) {
    func_->metaAttributes_.set("shapeInferer", inferer).check();
  }

  template <typename T>
  void setShapeInferer(std::function<Error(std::vector<topology::TensorPtr>&,
                                           std::vector<topology::TensorPtr>&,
                                           const T&)> inferer) {
    paddle::topology::meta::FunctionMetaPtr func = func_;
    func_->metaAttributes_
        .set<TensorShapeInfererWithAttrs>(
            "shapeInferer",
            [inferer, func](std::vector<topology::TensorPtr>& in,
                            std::vector<topology::TensorPtr>& out,
                            const topology::AttributeMap& attrs) {
              T val;
              auto err = func->parseAttribute(attrs, &val);
              if (!err.isOK()) return err;
              return inferer(in, out, val);
            })
        .check();
  }

  template <FunctionTensorType type>
  topology::meta::TensorMetaPtr& addTensor(
      size_t dim,
      int argType = -1,
      const Set<int>& dataTypes = {topology::DataType::DENSE},
      const Set<int>& seqTypes = topology::meta::DefaultSequenceType) {
    auto& meta = func_->addTensor<type>();
    meta->supportDataTypes(dataTypes)
        .supportSequenceTypes(seqTypes)
        .setShapeDimension(dim);
    if (argType != -1) meta->supportArgType(argType);
    return meta;
  }

  template <FunctionTensorType type>
  topology::meta::TensorMetaPtr& addTensor(
      const std::vector<size_t>& shape,
      int argType = -1,
      const Set<int>& dataTypes = {topology::DataType::DENSE},
      const Set<int>& seqTypes = topology::meta::DefaultSequenceType) {
    topology::meta::TensorMetaPtr& meta = func_->addTensor<type>();
    meta->supportDataTypes(dataTypes).supportSequenceTypes(seqTypes);
    meta->setShape(shape);
    if (argType != -1) meta->supportArgType(argType);
    return meta;
  }

  template <DeviceType devType>
  paddle::Error registerKernel(
      std::function<Error(const BufferArgs& ins,
                          const BufferArgs& outs,
                          const AttributeType& attrs)> kernel) {
    auto meta = func_;
    auto key = devType == DEVICE_TYPE_CPU ? "CPUKernel" : "GPUKernel";

    details::FunctionWithAttrs fn = [kernel, meta](
        const BufferArgs& ins,
        const BufferArgs& outs,
        const topology::AttributeMap& attrs) {
      AttributeType tmp;
      auto err = meta->parseAttribute(attrs, &tmp);
      if (!err.isOK()) return err;
      return kernel(ins, outs, tmp);
    };
    return func_->metaAttributes_.set(key, fn);
  }

  paddle::topology::meta::FunctionMetaPtr func_;
};

#define REGISTER_FUNCTION_META_INFO(CLASS) \
  static paddle::InitFunction __init_##CLASS##_instance__([] { CLASS inst; });

#define BEGIN_REGISTER_FUNCTION(cls, __func__, __attr_type__)            \
  class cls : public paddle::function::MetaInfoRegister<__attr_type__> { \
  public:                                                                \
    cls() : paddle::function::MetaInfoRegister<__attr_type__>(#cls) {    \
      registerKernel<DEVICE_TYPE_CPU>(__func__<DEVICE_TYPE_CPU>);        \
      registerKernel<DEVICE_TYPE_GPU>(__func__<DEVICE_TYPE_GPU>);

#define END_REGISTER_FUNCTION(cls) \
  }                                \
  }                                \
  ;                                \
  REGISTER_FUNCTION_META_INFO(cls)

}  // namespace function
}  // namespace paddle
