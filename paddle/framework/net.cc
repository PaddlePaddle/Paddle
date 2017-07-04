#include "paddle/framework/net.h"

namespace paddle {
namespace framework {

PlainNet::PlainNet(const NetDesc& def) {}

virtual Error PlainNet::InferShape() {
  for (auto& op : ops_) {
    // wrong shape
    auto err = op.InferShape();
    if (!err) return err;
  }
  // ok
  return Error();
}

virtual Error PlainNet::Run(Scope* scope = nullptr,
                            OpContext* context = nullptr, OpIndex begin = -1,
                            OpIndex end = -1) const {}

}  // namespace framework
}  // namespace paddle
