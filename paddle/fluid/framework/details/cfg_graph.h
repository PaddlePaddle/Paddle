#pragma once
#include <unordered_map>
#include <string>

#include "paddle/fluid/framework/"


namespace paddle {
namespace framework {
namespace details {

class ControlFlowGraph : public ir::Graph {
private:
  std::unordered_map<std::string>
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
