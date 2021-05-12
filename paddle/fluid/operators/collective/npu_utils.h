#pragma once

#include <string>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/memory/memory.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {
bool found_inf_data(const framework::ExecutionContext& ctx, 
        aclrtStream stream, 
        const Tensor* );

template<typename T>
bool fill_inf_data(ctx, aclrtStream stream, Tensor* out){
  T inf= static_cast<T>(std::numeric_limits<float>::infinity());
  auto out_dims=framework::vectorize(out->dims())

  out_var->mutable_data<T>(shape, place);
  auto runner = NpuOpRunner("FillV2", 
          {"dims", out_dims}, 
          {out}, 
          {{"value:", inf}});
  runner.Run(stream);
}

};
};

