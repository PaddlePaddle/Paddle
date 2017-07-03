#pragma once

#include <functional>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include "paddle/framework/op_base.h"
#include "paddle/framework/op_desc.pb.h"
#include "paddle/framework/op_proto.pb.h"

namespace paddle {
namespace framework {

class OpAttrLimit {
  // TODO
}

class OpProtoMaker {
 public:
  OpProtoMaker(OpProto* proto) : proto_(proto) {}

 protected:
  void AddInput(const std::string& name, const std::string& comment);
  void AddOutput(const std::string& name, const std::string& comment);
  AttrProto* AddAttr(const std::string& name, const std::string& comment,
                     const std::type_info& type_info);
  void AddType(const std::string& op_type);
  void AddComment(const std::string& comment);

  OpProto* proto_;
};

class OpRegistry {
 public:
  using OpCreator = std::function<OperatorBase*(OpDesc& op_desc)>;

  template <typename OpType, typename ProtoMakerType>
  static void RegisterOp(const std::string& op_type) {
    creators_[op_type] = [](const OpDesc& op_desc) {
      return new OpType(op_desc);
    };
    OpProto& op_proto = protos_[op_type];
    ProtoMakerType(&op_proto);
  }

  static std::unordered_map<std::string, OpCreator> creators_;
  static std::unordered_map<std::string, OpProto> protos_;
  static std::unordered_map<std::string, OpAttrLimit> attr_limits_;
};

template <typename OpType, typename ProtoMakerType>
class OpRegisterHelper {
 public:
  OpRegisterHelper(std::string op_type) {
    OpRegistry::RegisterOp<OpType, ProtoMakerType>(op_type);
  }
};

#define REGISTER_OP(op_class, op_maker_class, op_type)             \
  class op_class##Register {                                       \
   private:                                                        \
    const static OpRegisterHelper<#op_class, #op_maker_class> reg; \
  };                                                               \
  const Register op_class##Register::reg(#op_type);

class CosineOp {
  // ...
}

class CosineOpProtoMaker : public OpProtoMaker {
 public:
  CosineOpProtoMaker(OpProto* proto) : OpProtoMaker(proto) {
    AddInput("input", "input of cosine op");
    AddOutput("output", "output of cosine op");
    AddAttr("scale", "scale of cosine op", typeid(float))
        .Default(1.0)
        .LargerThan(0.0);
    AddType("cos");
    AddComment("This is cos op");
  }
}

REGISTER_OP(CosineOp, CosineOpProtoMaker, "cos");

}  // namespace framework
}  // namespace paddle
