#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/framework/framework.pb.h"

namespace paddle {
namespace inference {
namespace analysis {

template <>
void SetAttr<std::string>(framework::proto::OpDesc *op, const std::string &name,
                          const std::string &data) {
  auto *attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::STRING);
  attr->set_s(data);
}
template <>
void SetAttr<int>(framework::proto::OpDesc *op, const std::string &name,
                  const int &data) {
  auto *attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::INT);
  attr->set_i(data);
}
template <>
void SetAttr<int64_t>(framework::proto::OpDesc *op, const std::string &name,
                      const int64_t &data) {
  auto *attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::LONG);
  attr->set_l(data);
}
template <>
void SetAttr<std::vector<std::string>>(framework::proto::OpDesc *op,
                                       const std::string &name,
                                       const std::vector<std::string> &data) {
  auto *attr = op->add_attrs();
  attr->set_name(name);
  attr->set_type(paddle::framework::proto::AttrType::STRINGS);
  for (const auto &s : data) {
    attr->add_strings(s.c_str());
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
