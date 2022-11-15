/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/phi/core/macros.h"
#include "paddle/utils/any.h"

namespace paddle {
namespace framework {
namespace details {
using ProgramDescs = std::vector<ProgramDesc>;
constexpr char kProgramDescs[] = "program_descs";
constexpr char kStartupProgramDescs[] = "startup_program_descs";
}  // namespace details

namespace ir {
class Graph;

template <typename PassType>
struct PassRegistrar;

typedef std::unordered_set<std::string> PassRecorder;
constexpr char kPassRecorder[] = "pass_recorder";
constexpr char kEmbEltwiseLayernormPass[] =
    "embedding_eltwise_layernorm_fuse_pass_flag";
constexpr char kMultiheadMatmulPass[] = "multihead_matmul_fuse_pass_flag";
constexpr char kFusedMultiTransformerEncoderPass[] =
    "fused_multi_transformer_encoder_pass_flag";
constexpr char kFusedMultiTransformerDecoderPass[] =
    "fused_multi_transformer_decoder_pass_flag";
constexpr char kFusedMultiTransformerEncoderFuseQKVPass[] =
    "fused_multi_transformer_encoder_fuse_qkv_pass_flag";
constexpr char kFusedMultiTransformerDecoderFuseQKVPass[] =
    "fused_multi_transformer_decoder_fuse_qkv_pass_flag";
constexpr char kMultiDevicesFusedMultiTransformerEncoderFuseQKVPass[] =
    "multi_devices_fused_multi_transformer_encoder_fuse_qkv_pass_flag";
constexpr char kMultiDevicesFusedMultiTransformerDecoderFuseQKVPass[] =
    "multi_devices_fused_multi_transformer_decoder_fuse_qkv_pass_flag";
constexpr char kFusedMultiTransformerEncoderFusionCount[] =
    "fused_multi_transformer_encoder_fusion_count";
constexpr char kFusedMultiTransformerDecoderFusionCount[] =
    "fused_multi_transformer_decoder_fusion_count";
constexpr char kPrelnEmbEltwiseLayernormPass[] =
    "preln_embedding_eltwise_layernorm_fuse_pass_flag";

class Pass {
 public:
  Pass() = default;
  virtual ~Pass() {
    for (auto &attr : attrs_) {
      if (attr_dels_.find(attr.first) != attr_dels_.end()) {
        attr_dels_[attr.first]();
      }
    }
    attrs_.clear();
    attr_dels_.clear();
  }

  std::string Type() const { return type_; }

  Graph *Apply(Graph *graph) const;

  // Get a reference to the attributed previously set.
  template <typename AttrType>
  AttrType &Get(const std::string &attr_name) const {
    PADDLE_ENFORCE_NE(attrs_.find(attr_name),
                      attrs_.end(),
                      platform::errors::InvalidArgument(
                          "Attribute %s not registered for pass.", attr_name));
    try {
      return *paddle::any_cast<AttrType *>(attrs_.at(attr_name));
    } catch (paddle::bad_any_cast &) {
      auto TypeToString = [](const std::type_info &info) -> std::string {
        if (std::type_index(info) == std::type_index(typeid(bool *))) {
          return "bool";
        } else if (std::type_index(info) == std::type_index(typeid(int *))) {
          return "int";
        } else if (std::type_index(info) ==
                   std::type_index(typeid(const int *))) {
          return "const int";
        } else if (std::type_index(info) ==
                   std::type_index(typeid(std::string *))) {
          return "std::string";
        }
        return info.name();
      };

      PADDLE_THROW(platform::errors::InvalidArgument(
          "Invalid type for attritube %s, expected: %s, actual: %s.",
          attr_name,
          TypeToString(typeid(AttrType *)),
          TypeToString(attrs_.at(attr_name).type())));
    }
  }

  bool Has(const std::string &attr_name) const {
    return attrs_.count(attr_name) > 0;
  }

  void Erase(const std::string &attr_name) {
    if (!Has(attr_name)) {
      return;
    }
    if (attr_dels_.find(attr_name) != attr_dels_.end()) {
      attr_dels_[attr_name]();
      attr_dels_.erase(attr_name);
    }
    attrs_.erase(attr_name);
  }

  // Set a pointer to the attribute. Pass takes ownership of the attribute.
  template <typename AttrType>
  void Set(const std::string &attr_name, AttrType *attr) {
    if (default_pass_attrs_.count(attr_name) == 0) {
      PADDLE_ENFORCE_EQ(
          attrs_.count(attr_name),
          0,
          platform::errors::AlreadyExists(
              "Attribute %s already set in the pass.", attr_name));
    } else {
      VLOG(3) << "Setting the attribute " << attr_name << " for the pass "
              << type_;
    }
    attrs_[attr_name] = attr;
    attr_dels_[attr_name] = [attr, attr_name]() {
      VLOG(3) << "deleting " << attr_name;
      delete attr;
    };
  }

  // Set a pointer to the attribute. Pass doesn't take ownership. Caller
  // should delete the attribute.
  template <typename AttrType>
  void SetNotOwned(const std::string &attr_name, AttrType *attr) {
    PADDLE_ENFORCE_EQ(attrs_.count(attr_name),
                      0,
                      platform::errors::AlreadyExists(
                          "Attribute %s already set in the pass.", attr_name));
    attrs_[attr_name] = attr;
  }

  static void ApplyPassesToProgram(const std::vector<const Pass *> &passes,
                                   ProgramDesc *main_program,
                                   ProgramDesc *startup_program);

  virtual bool SupportApplyProgramViaGraph() const { return true; }

 protected:
  virtual void ApplyImpl(Graph *graph) const {
    PADDLE_THROW(platform::errors::Unimplemented(
        "The virtual pass called is not implemented."));
  }

  virtual void ApplyImpl(ProgramDesc *main_program,
                         ProgramDesc *startup_program) const;

  static void ConvertToPrograms(ir::Graph *graph,
                                ProgramDesc *main_program,
                                ProgramDesc *startup_program);

  // Some Pass must be placed before this Pass, and some
  // Pass must be placed after this Pass.
  virtual void CheckPrevPass() const {}

 private:
  template <typename PassType>
  friend struct PassRegistrar;

  void RegisterRequiredPassAttrs(const std::unordered_set<std::string> &attrs) {
    required_pass_attrs_.insert(attrs.begin(), attrs.end());
  }

  void RegisterRequiredGraphAttrs(
      const std::unordered_set<std::string> &attrs) {
    required_graph_attrs_.insert(attrs.begin(), attrs.end());
  }

  // Pass doesn't take ownership. PassRegistrar should delete default_attrs
  void RegisterDefaultPassAttrs(
      std::map<std::string, paddle::any> default_attr_values) {
    for (auto const &attr_name : default_attr_values) {
      default_pass_attrs_.insert(attr_name.first);
    }
    attrs_.insert(default_attr_values.begin(), default_attr_values.end());
  }

  void RegisterType(const std::string &type) { type_ = type; }

  mutable bool applied_{false};
  std::string type_;
  std::unordered_set<std::string> required_pass_attrs_;
  std::unordered_set<std::string> default_pass_attrs_;
  std::unordered_set<std::string> required_graph_attrs_;
  std::map<std::string, paddle::any> attrs_;
  std::map<std::string, std::function<void(void)>> attr_dels_;
};

using PassCreator = std::function<std::unique_ptr<Pass>()>;

class Registrar {
 public:
  // In our design, various kinds of passes,
  // have their corresponding registry and registrar. The action of
  // registration is in the constructor of a global registrar variable, which
  // are not used in the code that calls package framework, and would
  // be removed from the generated binary file by the linker. To avoid such
  // removal, we add Touch to all registrar classes and make USE_PASS macros to
  // call this method. So, as long as the callee code calls USE_PASS, the global
  // registrar variable won't be removed by the linker.
  void Touch() {}
};

class PassRegistry {
 public:
  static PassRegistry &Instance();

  bool Has(const std::string &pass_type) const {
    return map_.find(pass_type) != map_.end();
  }

  void Insert(const std::string &pass_type, const PassCreator &pass_creator) {
    PADDLE_ENFORCE_NE(Has(pass_type),
                      true,
                      platform::errors::AlreadyExists(
                          "Pass %s has been registered.", pass_type));
    map_.insert({pass_type, pass_creator});
  }

  std::unique_ptr<Pass> Get(const std::string &pass_type) const {
    if (pass_type == "tensorrt_subgraph_pass") {
      PADDLE_ENFORCE_EQ(Has(pass_type),
                        true,
                        platform::errors::InvalidArgument(
                            "Pass %s has not been registered. Please "
                            "use the paddle inference library "
                            "compiled with tensorrt or disable "
                            "the tensorrt engine in inference configuration! ",
                            pass_type));
    } else {
      PADDLE_ENFORCE_EQ(Has(pass_type),
                        true,
                        platform::errors::InvalidArgument(
                            "Pass %s has not been registered.", pass_type));
    }
    return map_.at(pass_type)();
  }

 private:
  PassRegistry() = default;
  std::unordered_map<std::string, PassCreator> map_;

  DISABLE_COPY_AND_ASSIGN(PassRegistry);
};

template <typename PassType>
struct PassRegistrar : public Registrar {
  explicit PassRegistrar(const char *pass_type) {
    PADDLE_ENFORCE_EQ(
        PassRegistry::Instance().Has(pass_type),
        false,
        platform::errors::AlreadyExists(
            "Pass '%s' is registered more than once.", pass_type));
    PassRegistry::Instance().Insert(
        pass_type, [this, pass_type]() -> std::unique_ptr<Pass> {
          std::unique_ptr<Pass> pass(new PassType());
          pass->RegisterRequiredPassAttrs(this->required_pass_attrs_);
          pass->RegisterRequiredGraphAttrs(this->required_graph_attrs_);
          pass->RegisterDefaultPassAttrs(this->default_attr_values_);
          pass->RegisterType(pass_type);
          return pass;
        });
  }

  ~PassRegistrar() {
    for (auto &attr : default_attr_values_) {
      if (default_attr_dels_.find(attr.first) != default_attr_dels_.end()) {
        default_attr_dels_[attr.first]();
      }
    }
    default_attr_values_.clear();
    default_attr_dels_.clear();
  }

  PassRegistrar<PassType> &RequirePassAttr(const std::string &attr) {
    required_pass_attrs_.insert(attr);
    return *this;
  }

  // PassRegistrar takes ownership of default_attr_value
  template <typename AttrType>
  PassRegistrar<PassType> &DefaultPassAttr(const std::string &attr,
                                           AttrType &&default_attr_value) {
    default_attr_values_[attr] = default_attr_value;
    default_attr_dels_[attr] = [default_attr_value, attr]() {
      delete default_attr_value;
    };
    return *this;
  }

  PassRegistrar<PassType> &RequireGraphAttr(const std::string &attr) {
    required_graph_attrs_.insert(attr);
    return *this;
  }

 private:
  std::unordered_set<std::string> required_pass_attrs_;
  std::unordered_set<std::string> required_graph_attrs_;
  std::map<std::string, paddle::any> default_attr_values_;
  std::map<std::string, std::function<void(void)>> default_attr_dels_;
};

#define STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(uniq_name, msg)                   \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

// Register a new pass that can be applied on the IR.
#define REGISTER_PASS(pass_type, pass_class)                \
  STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(                      \
      __reg_pass__##pass_type,                              \
      "REGISTER_PASS must be called in global namespace");  \
  static ::paddle::framework::ir::PassRegistrar<pass_class> \
      __pass_registrar_##pass_type##__(#pass_type);         \
  int TouchPassRegistrar_##pass_type() {                    \
    __pass_registrar_##pass_type##__.Touch();               \
    return 0;                                               \
  }                                                         \
  static ::paddle::framework::ir::PassRegistrar<pass_class> \
      &__pass_tmp_registrar_##pass_type##__ UNUSED =        \
          __pass_registrar_##pass_type##__

#define USE_PASS(pass_type)                           \
  STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(                \
      __use_pass_itself_##pass_type,                  \
      "USE_PASS must be called in global namespace"); \
  extern int TouchPassRegistrar_##pass_type();        \
  static int use_pass_itself_##pass_type##_ UNUSED =  \
      TouchPassRegistrar_##pass_type()

}  // namespace ir
}  // namespace framework
}  // namespace paddle
