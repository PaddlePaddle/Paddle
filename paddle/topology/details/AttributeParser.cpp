#include "AttributeParser.h"
#include "../Attribute.h"
#include "../AttributeMap.h"
#include "../meta/FunctionMeta.h"
namespace paddle {
namespace topology {
namespace details {

Error AttributeParser::operator()(Attribute *instance,
                                  const AttributeMap &attrs) const {
  for (auto it = attrs.begin(); it != attrs.end(); ++it) {
    auto &name = it->first;
    auto parserIt = this->find(name);
    if (parserIt == end()) {
      return Error("Cannot found parser %s", name.c_str());
    }
    auto err = parserIt->second(&it->second, instance);
    if (!err.isOK()) return err;
  }
  return Error();
}

AttributeParserPtr gCurParser = nullptr;
meta::FunctionMeta *gCurFuncMeta = nullptr;

FunctionMetaScope::~FunctionMetaScope() {
  auto parser = gCurParser;
  std::function<Error(const AttributeMap &, paddle::topology::Attribute *)>
      parserFunction =
          [parser](const AttributeMap &attrs,
                   paddle::topology::Attribute *instance) -> Error {
    return (*parser)(instance, attrs);
  };
  gCurFuncMeta->metaAttributes_.set("attribute_parser", parserFunction, false);
  gCurFuncMeta = nullptr;
}

}  // namespace details
}  // namespace topology
}  // namespace paddle
