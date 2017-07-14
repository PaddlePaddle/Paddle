#pragma once

#include <algorithm>
#include <type_traits>
#include "paddle/framework/attr_checker.h"
#include "paddle/framework/op_desc.pb.h"
#include "paddle/framework/op_proto.pb.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace framework {

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

  void AddComment(const std::string& comment) {
    *(proto_->mutable_comment()) = comment;
  }

  OpProto* proto_;
  OpAttrChecker* op_checker_;
};

class OpRegistry {
  using OpCreator = std::function<OperatorBase*()>;

 public:
  template <typename OpType, typename ProtoMakerType>
  static void RegisterOp(const std::string& op_type) {
    creators()[op_type] = [] { return new OpType; };
    OpProto& op_proto = protos()[op_type];
    OpAttrChecker& op_checker = op_checkers()[op_type];
    ProtoMakerType(&op_proto, &op_checker);
    *op_proto.mutable_type() = op_type;
    PADDLE_ENFORCE(
        op_proto.IsInitialized(),
        "Fail to initialize %s's OpProto, because %s is not initialized",
        op_type, op_proto.InitializationErrorString());
  }

  static OperatorPtr CreateOp(const OpDesc& op_desc) {
    std::string op_type = op_desc.type();
    OperatorPtr op(creators().at(op_type)());
    op->desc_ = op_desc;
    op->inputs_.reserve((size_t)op_desc.inputs_size());
    std::copy(op_desc.inputs().begin(), op_desc.inputs().end(),
              std::back_inserter(op->inputs_));
    op->outputs_.reserve((size_t)op_desc.outputs_size());
    std::copy(op_desc.outputs().begin(), op_desc.outputs().end(),
              std::back_inserter(op->outputs_));
    for (auto& attr : op_desc.attrs()) {
      op->attrs_[attr.name()] = AttrTypeHelper::GetAttrValue(attr);
    }
    op_checkers().at(op_type).Check(op->attrs_);
    op->Init();
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
  OpRegisterHelper(const char* op_type) {
    OpRegistry::RegisterOp<OpType, ProtoMakerType>(op_type);
  }
};

#define STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                        \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define REGISTER_OP(__op_type, __op_class, __op_maker_class)                 \
  STATIC_ASSERT_GLOBAL_NAMESPACE(__reg_op__##__op_type,                      \
                                 "REGISTER_OP must be in global namespace"); \
  static ::paddle::framework::OpRegisterHelper<__op_class, __op_maker_class> \
      __op_register_##__op_type##__(#__op_type);                             \
  int __op_register_##__op_type##_handle__() { return 0; }

#define REGISTER_OP_KERNEL(type, GPU_OR_CPU, PlaceType, KernelType)       \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      __reg_op_kernel_##type##_##GPU_OR_CPU##__,                          \
      "REGISTER_OP_KERNEL must be in global namespace");                  \
  struct __op_kernel_register__##type##__ {                               \
    __op_kernel_register__##type##__() {                                  \
      ::paddle::framework::OperatorWithKernel::OpKernelKey key;           \
      key.place_ = PlaceType();                                           \
      ::paddle::framework::OperatorWithKernel::AllOpKernels()[#type][key] \
          .reset(new KernelType());                                       \
    }                                                                     \
  };                                                                      \
  static __op_kernel_register__##type##__ __reg_kernel_##type##__;        \
  int __op_kernel_register_##type##_handle_##GPU_OR_CPU##__() { return 0; }

#define REGISTER_OP_GPU_KERNEL(type, KernelType) \
  REGISTER_OP_KERNEL(type, GPU, ::paddle::platform::GPUPlace, KernelType)

#define REGISTER_OP_CPU_KERNEL(type, KernelType) \
  REGISTER_OP_KERNEL(type, CPU, ::paddle::platform::CPUPlace, KernelType)

#define USE_OP_WITHOUT_KERNEL(op_type)                      \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                           \
      __use_op_without_kernel_##op_type,                    \
      "USE_OP_WITHOUT_KERNEL must be in global namespace"); \
  extern int __op_register_##op_type##_handle__();          \
  static int __use_op_ptr_##op_type##_without_kernel__      \
      __attribute__((unused)) = __op_register_##op_type##_handle__()

#define USE_OP_KERNEL(op_type, DEVICE_TYPE)                               \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      __use_op_kernel_##op_type##_##DEVICE_TYPE##__,                      \
      "USE_OP_KERNEL must be in global namespace");                       \
  extern int __op_kernel_register_##op_type##_handle_##DEVICE_TYPE##__(); \
  static int __use_op_ptr_##op_type##_##DEVICE_TYPE##_kernel__            \
      __attribute__((unused)) =                                           \
          __op_kernel_register_##op_type##_handle_##DEVICE_TYPE##__()

#ifdef PADDLE_ONLY_CPU
#define USE_OP(op_type)           \
  USE_OP_WITHOUT_KERNEL(op_type); \
  USE_OP_KERNEL(op_type, CPU);

#else
#define USE_OP(op_type)           \
  USE_OP_WITHOUT_KERNEL(op_type); \
  USE_OP_KERNEL(op_type, CPU);    \
  USE_OP_KERNEL(op_type, GPU)
#endif

}  // namespace framework
}  // namespace paddle
