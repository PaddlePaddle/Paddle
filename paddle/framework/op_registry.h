#pragma once

#include "paddle/framework/attr_checker.h"

//#include "paddle/framework/op_base.h"
#include <algorithm>
#include "paddle/framework/op_desc.pb.h"
#include "paddle/framework/op_proto.pb.h"

namespace paddle {
namespace framework {

//==================For test================//
class OpBase {
 public:
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  AttributeMap attr_map_;

  virtual std::string Run() const = 0;
  virtual ~OpBase() {}
};
//=========================================//

// helper class to set attribute type
struct AttrTypeHelper {
  template <typename T>
  static void SetAttrType(AttrProto* attr);

  static Attribute GetAttrValue(const AttrDesc& attr_desc) {
    switch (attr_desc.type()) {
      case paddle::framework::AttrType::INT: {
        return attr_desc.i();
      }
      case paddle::framework::AttrType::FLOAT: {
        return attr_desc.f();
      }
      case paddle::framework::AttrType::STRING: {
        return attr_desc.s();
      }
      case paddle::framework::AttrType::INTS: {
        std::vector<int> val(attr_desc.ints_size());
        for (int i = 0; i < attr_desc.ints_size(); ++i) {
          val[i] = attr_desc.ints(i);
        }
        return val;
      }
      case paddle::framework::AttrType::FLOATS: {
        std::vector<float> val(attr_desc.floats_size());
        for (int i = 0; i < attr_desc.floats_size(); ++i) {
          val[i] = attr_desc.floats(i);
        }
        return val;
      }
      case paddle::framework::AttrType::STRINGS: {
        std::vector<std::string> val(attr_desc.strings_size());
        for (int i = 0; i < attr_desc.strings_size(); ++i) {
          val[i] = attr_desc.strings(i);
        }
        return val;
      }
    }
    PADDLE_ENFORCE(false, "Unknown OpDesc::AttrDesc::type !");
    return boost::blank();
  }
};

// this class not only make proto but also init attribute checkers.
class OpProtoAndCheckerMaker {
 public:
  OpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : proto_(proto), op_checker_(op_checker) {}

 protected:
  void AddInput(const std::string& name, const std::string& comment) {
    auto input = proto_->mutable_inputs()->Add();
    *input->mutable_name() = name;
    *input->mutable_comment() = comment;
  }

  void AddOutput(const std::string& name, const std::string& comment) {
    auto output = proto_->mutable_outputs()->Add();
    *output->mutable_name() = name;
    *output->mutable_comment() = comment;
  }

  template <typename T>
  TypedAttrChecker<T>& AddAttr(const std::string& name,
                               const std::string& comment) {
    auto attr = proto_->mutable_attrs()->Add();
    *attr->mutable_name() = name;
    *attr->mutable_comment() = comment;
    AttrTypeHelper::SetAttrType<T>(attr);
    return op_checker_->AddAttrChecker<T>(name);
  }

  void AddType(const std::string& op_type) { proto_->set_type(op_type); }

  void AddComment(const std::string& comment) {
    *(proto_->mutable_comment()) = comment;
  }

  OpProto* proto_;
  OpAttrChecker* op_checker_;
};

class OpRegistry {
  using OpCreator = std::function<OpBase*()>;

 public:
  template <typename OpType, typename ProtoMakerType>
  static void RegisterOp(const std::string& op_type) {
    creators()[op_type] = [] { return new OpType; };
    OpProto& op_proto = protos()[op_type];
    OpAttrChecker& op_checker = op_checkers()[op_type];
    ProtoMakerType(&op_proto, &op_checker);
    PADDLE_ENFORCE(op_proto.IsInitialized(),
                   "Fail to initialize %s's OpProto !", op_type);
  }

  static OpBase* CreateOp(const OpDesc& op_desc) {
    std::string op_type = op_desc.type();
    OpBase* op = creators().at(op_type)();
    op->inputs_.reserve((size_t)op_desc.inputs_size());
    std::copy(op_desc.inputs().begin(), op_desc.inputs().end(),
              std::back_inserter(op->inputs_));
    op->outputs_.reserve((size_t)op_desc.outputs_size());
    std::copy(op_desc.outputs().begin(), op_desc.outputs().end(),
              std::back_inserter(op->outputs_));
    for (auto& attr : op_desc.attrs()) {
      op->attr_map_[attr.name()] = AttrTypeHelper::GetAttrValue(attr);
    }
    op_checkers().at(op_type).Check(op->attr_map_);
    return op;
  }

 private:
  static std::unordered_map<std::string, OpCreator>& creators() {
    static std::unordered_map<std::string, OpCreator> creators_;
    return creators_;
  }

  static std::unordered_map<std::string, OpProto>& protos() {
    static std::unordered_map<std::string, OpProto> protos_;
    return protos_;
  };

  static std::unordered_map<std::string, OpAttrChecker>& op_checkers() {
    static std::unordered_map<std::string, OpAttrChecker> op_checkers_;
    return op_checkers_;
  };
};

template <typename OpType, typename ProtoMakerType>
class OpRegisterHelper {
 public:
  OpRegisterHelper(std::string op_type) {
    OpRegistry::RegisterOp<OpType, ProtoMakerType>(op_type);
  }
};

#define REGISTER_OP(__op_class, __op_maker_class, __op_type)         \
  class __op_class##Register {                                       \
   private:                                                          \
    const static OpRegisterHelper<__op_class, __op_maker_class> reg; \
  };                                                                 \
  const OpRegisterHelper<__op_class, __op_maker_class>               \
      __op_class##Register::reg(#__op_type);

}  // namespace framework
}  // namespace paddle
