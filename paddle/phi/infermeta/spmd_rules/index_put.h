#pragma once

#include <vector>

#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

SpmdInfo IndexPutInferSpmd(const DistMetaTensor& x,
                           const std::vector<DistMetaTensor>& indices,
                           const DistMetaTensor& value,
                           bool accumulate);

SpmdInfo IndexPutInferSpmdReverse(const DistMetaTensor& x,
                                  const std::vector<DistMetaTensor>& indices,
                                  const DistMetaTensor& value,
                                  const DistMetaTensor& out,
                                  bool accumulate);

}  // namespace distributed
}  // namespace phi