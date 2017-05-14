#include "Attribute.h"

namespace paddle {
namespace topology {

Attribute::~Attribute() {}
AttributeParserPtr Attribute::parser;
meta::FunctionMetaPtr Attribute::funcMeta;
}  // namespace topology
}  // namespace paddle
