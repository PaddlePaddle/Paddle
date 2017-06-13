#pragma once

#include <sstream>
#include <string>
#include <type_traits>
#include "OptimizerConfig.pb.h"
#include "paddle/utils/Logging.h"
#include "tensor.h"

namespace paddle {
namespace optimizer {

static unsigned CalStateSize() { return 0; }

template <typename HEAD, typename... TAIL>
unsigned CalStateSize(const HEAD& head, const TAIL&... tail) {
  return sizeof head + CalStateSize(tail...);
}

template <typename... TAIL>
unsigned CalStateSize(const Tensor* head, const TAIL&... tail) {
  return head->size() + CalStateSize(tail...);
}

static void TensorToProto(const Tensor& tensor, TensorProto* proto) {
  proto->set_data_type(TensorProto::PADDLE_ELEMENT_TYPE_FLOAT32);
  std::stringstream os;
  for (size_t i = 0; i < tensor.size(); ++i) {
    os << tensor[i];
    proto->add_content(os.str());
    os.clear();
  }
}

static void ProtoToTensor(const TensorProto& proto, Tensor* tensor) {
  std::stringstream sin;
  for (auto i = 0; i < proto.content_size(); ++i) {
    sin << proto.content(i);
    sin >> (*tensor)[i];
    sin.clear();
  }
}

}  // namespace optimizer
}  // namespace paddle
