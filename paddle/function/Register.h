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
  typedef topology::meta::FunctionMeta::TenserShapeInferer TenserShapeInferer;
  typedef topology::meta::FunctionTensorType FunctionTensorType;
  static constexpr FunctionTensorType INPUT =
      topology::meta::FunctionTensorType::INPUT;
  static constexpr FunctionTensorType OUTPUT =
      topology::meta::FunctionTensorType::OUTPUT;

  void setShapeInferer(TenserShapeInferer inferer) {
    func_->metaAttributes_.set("shapeInferer", inferer).check();
  }

  template <FunctionTensorType type>
  void addTensor(
      size_t dim,
      int argType = -1,
      const Set<int>& dataTypes = {topology::DataType::DENSE},
      const Set<int>& seqTypes = topology::meta::DefaultSequenceType) {
    auto& meta = func_->addTensor<type>();
    meta->supportDataTypes(dataTypes)
        .supportSequenceTypes(seqTypes)
        .setShapeDimension(dim);
    if (argType != -1) meta->supportArgType(argType);
  }

  template <FunctionTensorType type>
  void addTensor(
      const std::vector<size_t>& shape,
      int argType = -1,
      const Set<int>& dataTypes = {topology::DataType::DENSE},
      const Set<int>& seqTypes = topology::meta::DefaultSequenceType) {
    topology::meta::TensorMetaPtr& meta = func_->addTensor<type>();
    meta->supportDataTypes(dataTypes).supportSequenceTypes(seqTypes);
    meta->setShape(shape);
    if (argType != -1) meta->supportArgType(argType);
  }

  template <DeviceType devType>
  paddle::Error registerKernel(
      std::function<Error(const BufferArgs& ins,
                          const BufferArgs& outs,
                          const AttributeType& attrs)> kernel) {
    auto meta = func_;
    auto key = devType == DEVICE_TYPE_CPU ? "CPUKernel" : "GPUKernel";
    auto inited = std::make_shared<bool>(false);
    auto tmp = std::make_shared<AttributeType>();
    details::FunctionWithAttrs fn = [kernel, meta, inited, tmp](
        const BufferArgs& ins,
        const BufferArgs& outs,
        const topology::AttributeMap& attrs) {
      bool& init = *inited;
      if (!init) {
        auto parserFunction =
            meta->metaAttributes_.template get<std::function<Error(
                const topology::AttributeMap&, topology::Attribute*)>>(
                "attribute_parser");
        auto err = parserFunction(attrs, tmp.get());
        if (!err.isOK()) return err;
        init = true;
      }
      return kernel(ins, outs, *tmp);
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
