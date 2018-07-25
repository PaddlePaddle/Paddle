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
#include <string>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace framework {
namespace ir {

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

  virtual std::unique_ptr<Graph> Apply(std::unique_ptr<Graph> graph) const = 0;

  template <typename AttrType>
  AttrType &Get(const std::string &attr_name) const {
    return *boost::any_cast<AttrType *>(attrs_.at(attr_name));
  }

  template <typename AttrType>
  void Set(const std::string &attr_name, AttrType *attr) {
    PADDLE_ENFORCE(attrs_.count(attr_name) == 0);
    attrs_[attr_name] = attr;
    attr_dels_[attr_name] = [attr, attr_name]() {
      VLOG(3) << "deleting " << attr_name;
      delete attr;
    };
  }

  template <typename AttrType>
  void SetNotOwned(const std::string &attr_name, AttrType *attr) {
    PADDLE_ENFORCE(attrs_.count(attr_name) == 0);
    attrs_[attr_name] = attr;
  }

 private:
  std::map<std::string, boost::any> attrs_;
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

  void Insert(const std::string &type, const PassCreator &pass_creator) {
    PADDLE_ENFORCE(!Has(type), "Pass %s has been registered", type);
    map_.insert({type, pass_creator});
  }

  std::unique_ptr<Pass> Get(const std::string &type) const {
    PADDLE_ENFORCE(Has(type), "Pass %s has not been registered", type);
    return map_.at(type)();
  }

 private:
  PassRegistry() = default;
  std::unordered_map<std::string, PassCreator> map_;

  DISABLE_COPY_AND_ASSIGN(PassRegistry);
};

template <typename PassType>
struct PassRegistrar : public Registrar {
  explicit PassRegistrar(const char *pass_type) {
    PADDLE_ENFORCE(!PassRegistry::Instance().Has(pass_type),
                   "'%s' is registered more than once.", pass_type);
    PassRegistry::Instance().Insert(pass_type, []() -> std::unique_ptr<Pass> {
      return std::unique_ptr<Pass>(new PassType());
    });
  }
};

#define STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(uniq_name, msg)                   \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define REGISTER_PASS(pass_type, pass_class)                \
  STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(                      \
      __reg_pass__##pass_type,                              \
      "REGISTER_PASS must be called in global namespace");  \
  static ::paddle::framework::ir::PassRegistrar<pass_class> \
      __pass_registrar_##pass_type##__(#pass_type);         \
  int TouchPassRegistrar_##pass_type() {                    \
    __pass_registrar_##pass_type##__.Touch();               \
    return 0;                                               \
  }

#define USE_PASS(pass_type)                                           \
  STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(                                \
      __use_pass_itself_##pass_type,                                  \
      "USE_PASS must be called in global namespace");                 \
  extern int TouchPassRegistrar_##pass_type();                        \
  static int use_pass_itself_##pass_type##_ __attribute__((unused)) = \
      TouchPassRegistrar_##pass_type()

}  // namespace ir
}  // namespace framework
}  // namespace paddle
