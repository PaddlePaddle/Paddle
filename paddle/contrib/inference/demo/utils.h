#pragma once
#include <string>
#include <vector>

#include "paddle/contrib/inference/paddle_inference_api.h"

namespace paddle {
namespace demo {

static void split(const std::string& str,
                  char sep,
                  std::vector<std::string>* pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

/*
 * Get a summary of a PaddleTensor content.
 */
static std::string SummaryTensor(const PaddleTensor& tensor) {
  std::stringstream ss;
  int num_elems = tensor.data.length / PaddleDtypeSize(tensor.dtype);

  ss << "data[:10]\t";
  switch (tensor.dtype) {
    case PaddleDType::INT64: {
      for (int i = 0; i < std::min(num_elems, 10); i++) {
        ss << static_cast<int64_t*>(tensor.data.data)[i] << " ";
      }
      break;
    }
    case PaddleDType::FLOAT32:
      for (int i = 0; i < std::min(num_elems, 10); i++) {
        ss << static_cast<float*>(tensor.data.data)[i] << " ";
      }
      break;
  }
  return ss.str();
}

}  // namespace demo
}  // namespace paddle
