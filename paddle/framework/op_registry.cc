#include <paddle/framework/op_registry.h>

namespace paddle {
namespace framework {

template <>
void AttrTypeHelper::SetAttrType<int>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::INT);
}

template <>
void AttrTypeHelper::SetAttrType<float>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::FLOAT);
}

template <>
void AttrTypeHelper::SetAttrType<std::string>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::STRING);
}

template <>
void AttrTypeHelper::SetAttrType<std::vector<int>>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::INTS);
}

template <>
void AttrTypeHelper::SetAttrType<std::vector<float>>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::FLOATS);
}

template <>
void AttrTypeHelper::SetAttrType<std::vector<std::string>>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::STRINGS);
}
}  // namespace framework
}  // namespace paddle