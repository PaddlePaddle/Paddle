// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/paddle/pir/pass_manager_method_class.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace ap::paddle {

struct PirPassManagerMethodClass {
  using Self = ap::paddle::PassManager;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.shared_ptr().get();
    std::ostringstream ss;
    ss << "<PirPassManager object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> AddPass(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);

    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    if (args.at(0).template Has<adt::Nothing>()) {
      // Do Nothing
    } else if (args.at(0).template CastableTo<ap::paddle::Pass>()) {
      ADT_LET_CONST_REF(pass, args.at(0).template CastTo<ap::paddle::Pass>())
          << adt::errors::TypeError{std::string() +
                                    "PirPassManager.add_pass() failed. the "
                                    "argument 1 should be a PirPass (not a " +
                                    axpr::GetTypeName(args.at(0)) + ")."};
      ADT_CHECK(pass->pir_pass != nullptr)
          << adt::errors::TypeError{std::string() + "PirPass being used."};
      self->pir_pass_manager->AddPass(std::move(pass.shared_ptr()->pir_pass));
    } else if (args.at(0).template CastableTo<std::string>()) {
      ADT_LET_CONST_REF(pass_name, args.at(0).template CastTo<std::string>());
      try {
        auto pass = pir::PassRegistry::Instance().Get(pass_name);
        self->pir_pass_manager->AddPass(std::move(pass));
      } catch (const std::exception& e) {
        return adt::errors::InvalidArgumentError{
            std::string() + "no pass found. pass-name: " + pass_name};
      }
    } else {
      return adt::errors::NotImplementedError{
          std::string() +
          "the argument 1 of add_pass() should be a PirPass or str."};
    }
    return self_val;
  }

  static adt::Result<axpr::Value> Run(const axpr::Value& self_val,
                                      const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_LET_CONST_REF(program,
                      args.at(0).template CastTo<ap::paddle::Program>())
        << adt::errors::TypeError{std::string() +
                                  "PirPassManager.run() failed. the argument 1 "
                                  "should be a PirProgram (not a " +
                                  axpr::GetTypeName(args.at(0)) + ")."};
    self->pir_pass_manager->Run(program->pir_program.get());
    return adt::Nothing{};
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetPirPassManagerClass() {
  using Impl = PirPassManagerMethodClass;
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "PirPassManager", [&](const auto& Yield) {
        Yield("__str__", &Impl::ToString);
        Yield("add_pass", &Impl::AddPass);
        Yield("run", &Impl::Run);
      }));
  return axpr::MakeGlobalNaiveClassOps<ap::paddle::PassManager>(cls);
}

}  // namespace ap::paddle
