#include "paddle/framework/grad_op_creator.h"

namespace paddle {
namespace framework {

OperatorBase* GradOpCreator::Create() {
  BuildOpInOutArgList();
  OperatorBase* grad_op = OpRegistry::grad_creators().at(op_->type_)();
  CompleteGradOp(grad_op);
  return grad_op;
}

OpInOutArg* GradOpCreator::BuildArg(const VarProto& var,
                                    const VarIndexMap& var_map,
                                    const vector<int>& format, InOutType type) {
  int idx = var_map.at(var.name());
  int begin_idx = format.empty() ? idx : format.at(idx);
  int end_idx = format.empty() ? idx + 1 : format.at(idx + 1);
  return new OpInOutArg(var.name(), type, !var.ignore_gradient(), begin_idx,
                        end_idx);
}

void GradOpCreator::BuildOpInOutArgList() {
  const OpProto& op_proto = OpRegistry::protos().at(op_->type);
  const auto& var_map = *(OpRegistry::VarIndexMaps().at(op->type_));
  const vector<int>& in_format =
      op_->attrs_.count("input_format")
          ? op->GetAttr<std::vector<int>>("input_format")
          : std::vector<int>();
  const vector<int>& out_format =
      op_->attrs_.count("output_format")
          ? op->GetAttr<std::vector<int>>("output_format")
          : std::vector<int>();
  for (const auto& var : op_proto.inputs()) {
    arg_list_.emplace_back(
        std::shared_ptr<OpInOutArg>(BuildArg(var, var_map, in_format, IN)));
  }
  for (const auto& var : op_proto.outputs()) {
    arg_list_.emplace_back(
        std::shared_ptr<OpInOutArg>(BuildArg(var, var_map, out_format, OUT)));
  }
}

void GradOpCreator::PushArgIntoGradOp(const OpInOutArg* arg,
                                      vector<std::string>& in_out,
                                      vector<int>& format, VarIndexMap* varmap,
                                      int& idx, bool is_grad) {
  std::string var_name = arg->proto_name_;
  if (is_grad) {
    var_name += OperatorBase::GRAD_VAR_SUFFIX();
  }
  *(varmap)[var_name] = idx++;
  size_t pre_sz = in_out.size();
  auto base_it = arg->type == IN ? op_->inputs_.begin() : op_->outputs_.begin();
  std::copy(base_it + arg->begin_idx_, base_it + arg->end_idx_,
            std::back_inserter(in_out));
  if (is_grad) {
    for (size_t i = pre_sz; i < in_out.size(); ++i) {
      in_out[i] += OperatorBase::GRAD_VAR_SUFFIX();
    }
  }
  format.push_back(in_out.size());
}

void GradOpCreator::CompleteGradOp(OperatorBase* grad_op) const {
  grad_op->type_ = op_->type_ + "@GRAD";  // not necessary
  grad_op->attrs_ = op_->attrs_;
  grad_op->attrs_.erase("input_format");
  grad_op->attrs_.erase("output_format");
  VarIndexMap* grad_varmap = new VarIndexMap();
  int in_idx = 0;
  int out_idx = 0;
  vector<int> in_format({0});
  vector<int> out_format({0});
  for (const auto& arg : arg_list_) {
    // op_'s inputs_ and outputs_
    if (arg->needed_in_grad_) {
      PushArgIntoGradOp(arg.get(), grad_op->inputs_, in_format, grad_varmap,
                        in_idx, false);
    }
    if (arg->type_ == IN) {
      // gradients of op_'s inputs_
      PushArgIntoGradOp(arg.get(), grad_op->outputs_, out_format, grad_varmap,
                        out_idx, true);
    } else {
      // gradients of op_'s outputs_
      PushArgIntoGradOp(arg.get(), grad_op->inputs_, in_format, grad_varmap,
                        in_idx, true);
    }
  }
  grad_op->attrs_["input_format"] = in_format;
  grad_op->attrs_["output_format"] = out_format;
  grad_op->in_out_idxs_.reset(grad_varmap);
}

}  // namespace framework
}  // namespace paddle