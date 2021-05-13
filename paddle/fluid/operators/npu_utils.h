#pragma once

#include <string>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/memory/memory.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/hccl_helper.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#endif


namespace paddle {
namespace operators {
bool found_inf_data(const framework::ExecutionContext& ctx ,
        aclrtStream stream, const paddle::framework::Tensor* );
};
};

