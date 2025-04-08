#include "paddle/phi/infermeta/spmd_rules/argmin.h"
#include "paddle/phi/infermeta/spmd_rules/argmax.h"
#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
SpmdInfo ArgMinInferSpmdBase(const DistMetaTensor& x,
                             int axis,
                             bool keepdims,
                             bool flatten){
    return ArgMaxInferSpmdBase(x,axis,keepdims,flatten);
}

SpmdInfo ArgMinInferSpmdReverseBase(const DistMetaTensor& x,
                                    const DistMetaTensor& out,
                                    int axis,
                                    bool keepdims,
                                    bool flatten){
    return ArgMaxInferSpmdReverseBase(x,out,axis,keepdims,flatten);
}
SpmdInfo ArgMinInferSpmdDynamic(const DistMetaTensor& x,
                                const Scalar& axis,
                                bool keepdims,
                                bool flatten,
                                DataType dtype){
                                    return ArgMaxInferSpmdDynamic(x,axis,keepdims,flatten,dtype);
}

}