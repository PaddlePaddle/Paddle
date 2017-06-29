#pragma once

#include <vector>

#include "paddle/framework/variable.h"
#include "paddle/utils/Error.h"

namespace paddle {
namespace framework {

/**
 * following is just for demo
 */
class Context {};
class CpuContext: public Context {};
class GpuContext: public Context {};

class Scope {
 public:
  Variable* getOrCreateVariable(const std::string name);
};

class OpDesc {};

class AttrbuteMap {
 public:

  template<typename T>
  Error get(const std::string key, T* attr) const;
};

/// OperatorBase provide base element of an Operator without any template.
class OperatorBase {
 public:
  explicit OperatorBase(const OpDesc& desc);
  virtual ~OperatorBase() {}

  /// initialize Attributes of this OP from proto message desc.attrs()
  /// you should derive this function to init the attr you need in OP.
  virtual Error InitializeAttributes(const AttrbuteMap& attrs) = 0;
  virtual Error Run(Scope* scope, Context* context) const = 0;

 protected:
  std::string type_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

/// Operator is the class your should derive when implement a new Operator.
template <typename DeviceContext>
class Operator : public OperatorBase {
 public:
  explicit Operator(const OpDesc& desc): OperatorBase(desc) {}

 private:
  /// This function will get all input and output Vars from scope and ten call
  /// Run(std::vector<Variable> inputs, std::vector<Variable> outputs, T* context)
  Error Run(Scope* scope, Context* context) const final {
    DeviceContext* dev_context = dynamic_cast<DeviceContext*>(context);
    if (dev_context == nullptr) {
      return Error("dynamic_cast devContext failed!");
    }

    std::vector<Variable*> input_vars;
    std::vector<Variable*> output_vars;

    input_vars.reserve(inputs_.size());
    for(auto& input: inputs_) {
      input_vars.push_back(scope->getOrCreateVariable(input));
    }
    output_vars.reserve(outputs_.size());
    for(auto& input: outputs_) {
      output_vars.push_back(scope->getOrCreateVariable(input));
    }

    return Run(input_vars, output_vars, dev_context);
  }

  // when implement an Op, your should implement this function.
  virtual Error Run(std::vector<Variable*>& inputs,
                    std::vector<Variable*>& outputs,
                    DeviceContext* context) const = 0;
};

class Net {
 public:
  Error Run(Scope* scope, Context* context) {
    for (auto& op : operators_) {
      Error err = op->Run(scope, context);
      if (!err.isOK()) {
        return err;
      }
    }
    return Error();
  }

 private:
  std::vector<std::unique_ptr<OperatorBase>> operators_;
};

}  // namespace framework
}  // namespace paddle