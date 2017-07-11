#include "paddle/framework/net.h"

namespace paddle {
namespace framework {

PlainNet::PlainNet(const NetDesc& def) {}

void PlainNet::InferShape(Scope* scope) {
  for (auto& op : ops_) {
    op.InferShape();
  }
}

void PlainNet::Run(Scope* scope) const {}

}  // namespace framework
}  // namespace paddle
