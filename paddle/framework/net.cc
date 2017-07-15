#include "paddle/framework/net.h"

namespace paddle {
namespace framework {

void PlainNet::CompleteAddOp() {
  std::unordered_set<std::string> input_set;
  std::unordered_set<std::string> output_set;
  std::unordered_set<std::string> temp_output;
  for (auto& op : ops_) {
    for (auto& ipt : op->inputs_) {
      if (!Contains(output_set, ipt)) {  // Not other op's output
        input_set.insert(ipt);
      } else {
        temp_output.insert(ipt);
      }
    }

    for (auto& opt : op->outputs_) {
      output_set.insert(opt);
    }
  }
  inputs_.reserve(input_set.size());
  std::copy(input_set.begin(), input_set.end(), std::back_inserter(inputs_));

  outputs_.reserve(output_set.size());
  std::vector<int> tmp_index;
  tmp_index.reserve(temp_output.size());
  int idx = 0;
  for (auto& opt : output_set) {
    if (Contains(temp_output, opt)) {
      tmp_index.push_back(idx);
    }
    outputs_.push_back(opt);
    ++idx;
  }

  attrs_["temporary_index"] = tmp_index;
  add_op_done_ = true;
}

}  // namespace framework
}  // namespace paddle