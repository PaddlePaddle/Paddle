#pragma once
#include <string>

namespace paddle {
namespace framework {
struct VarId {
  explicit VarId(const std::string& name) : name(name), unique_id(-1) {}
  VarId(const std::string& name, int id) : name(name), unique_id(id) {}
  std::string name;
  /*default -1 if uninitialized*/
  int unique_id;
};

}  // namespace framework
}  // namespace paddle
