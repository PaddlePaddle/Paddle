

#include "paddle/pten/core/compat/op_utils.h"

namespace pten {

KernelSignature HistogramOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "histogram", {"X"}, {"bins", "min", "max"}, {"Out"});
}


}  // namespace pten

PT_REGISTER_ARG_MAPPING_FN(histogram, pten::HistogramOpArgumentMapping);