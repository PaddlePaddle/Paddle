
#pragma once

#include <vector>

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

SpmdInfo ArgMinInferSpmdBase(const DistMetaTensor& x,
                             int axis,
                             bool keepdims,
                             bool flatten);

SpmdInfo ArgMinInferSpmdReverseBase(const DistMetaTensor& x,
                                    const DistMetaTensor& out,
                                    int axis,
                                    bool keepdims,
                                    bool flatten);

SpmdInfo ArgMinInferSpmdDynamic(const DistMetaTensor& x,
                                const Scalar& axis,
                                bool keepdims,
                                bool flatten,
                                DataType dtype);

}  // namespace distributed
}  // namespace phi