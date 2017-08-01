#pragma once

#include "paddle/framework/op_proto.pb.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace framework {
class OpRegistry;

enum InOutType { IN, OUT };

struct OpInOutArg {
  OpInOutArg(const std::string& proto_name, const InOutType& type,
             bool needed_in_grad, size_t begin_idx, size_t end_idx)
      : proto_name_(proto_name),
        type_(type),
        needed_in_grad_(needed_in_grad),
        begin_idx_(begin_idx),
        end_idx_(end_idx) {}

  std::string proto_name_;
  InOutType type_;
  bool needed_in_grad_;
  size_t begin_idx_;
  size_t end_idx_;
};

class GradOpBuilder {
  using VarIndexMap = std::unordered_map<std::string, int>;

 public:
  GradOpBuilder(const OperatorBase& op) : op_(op) {}
  OperatorBase* Build();

 private:
  OpInOutArg* BuildArg(const VarProto& var, const VarIndexMap& var_map,
                       const std::vector<int>& format, InOutType type);
  void BuildOpInOutArgList();
  void AddArgIntoGradOp(const OpInOutArg* arg, std::vector<std::string>& in_out,
                        std::vector<int>& format, VarIndexMap* varmap, int& idx,
                        bool is_grad) const;
  void CompleteGradOp(OperatorBase* grad_op) const;
  const OperatorBase& op_;
  std::vector<std::shared_ptr<OpInOutArg>> arg_list_;
};

}  // namespace framework
}  // namespace paddle
