#include "paddle/infrt/dialect/pten/pten_base.h"

#include "paddle/infrt/common/global.h"

namespace infrt {
namespace pten {

AllocatorType AllocatorType::get(const std::string& kind) {
  return Base::get(::infrt::Global::getMLIRContext(), kind);
}

std::string& AllocatorType::kind() { return getImpl()->kind_; }

}  // namespace pten
}  // namespace infrt
