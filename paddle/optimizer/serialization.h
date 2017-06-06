#ifndef PADDLE_OPTIMIZER_SERIALIZARION_H
#define PADDLE_OPTIMIZER_SERIALIZARION_H

#include <sstream>
#include <string>
#include "OptimizerConfig.pb.h"
#include "paddle/utils/Logging.h"
#include "tensor.h"

namespace paddle {
namespace optimizer {

static void TensorToProto(const Tensor& tensor, TensorProto* proto) {
  proto->set_data_type(TensorProto::PADDLE_ELEMENT_TYPE_FLOAT32);
  proto->set_size(tensor.size());
  std::stringstream os;
  for (size_t i = 0; i < tensor.size(); ++i) {
    os << tensor[i];
    proto->add_content(os.str());
    os.clear();
  }
}

static void ProtoToTensor(const TensorProto& proto, Tensor* tensor) {
  CHECK(proto.size() == tensor->size()) << "unmatch shape of proto and tensor";
  std::stringstream sin;
  for (auto i = 0; i < proto.content_size(); ++i) {
    sin << proto.content(i);
    sin >> (*tensor)[i];
    sin.clear();
  }
}

}  // namespace optimizer
}  // namespace paddle
#endif
