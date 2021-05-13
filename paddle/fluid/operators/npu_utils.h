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
bool FoundNanOrInf(const framework::ExecutionContext& ctx ,
        aclrtStream stream, const paddle::framework::Tensor* );

template <typename T>
void FillNpuTensorWithConstant(Tensor *tensor, T val) {
  // NOTE(zhiqiu): we found that power sometimes returns 0 when val is small
  // like 1e-8.
  constexpr float MIN_PRECISION_FOR_POWER = 1e-3;
  PADDLE_ENFORCE_EQ(
      tensor->IsInitialized(), true,
      platform::errors::InvalidArgument("The tensor should be initialized."));
  PADDLE_ENFORCE_EQ(
      platform::is_npu_place(tensor->place()), true,
      platform::errors::InvalidArgument("The tensor should be on NPUPlace."));
  // do async for better performance
  if ((typeid(float) == typeid(T) || typeid(platform::float16) == typeid(T)) &&
      static_cast<float>(val) > MIN_PRECISION_FOR_POWER && std::isinf(val)) {
    Tensor tmp(tensor->type());
    tmp.Resize(tensor->dims());
    tmp.mutable_data<T>(tensor->place());
    auto stream = GetCurrentNPUStream(
        BOOST_GET_CONST(platform::NPUPlace, tensor->place()).device);
    platform::NPUMemsetAsync(tmp.data<void>(), 0, tmp.numel() * sizeof(T),
                             stream);
    auto runner = NpuOpRunner("Power", {tmp}, {*tensor},
                              {{"power", static_cast<float>(1)},
                               {"scale", static_cast<float>(0)},
                               {"shift", static_cast<float>(val)}});
    runner.Run(stream);
  } else {
    T *array = new T[tensor->numel()];
    for (unsigned int i = 0; i < tensor->numel(); ++i) {
      array[i] = static_cast<T>(val);
    }
    std::vector<T> vec(tensor->numel(), static_cast<T>(val));
    // do sync copy
    memory::Copy(BOOST_GET_CONST(platform::NPUPlace, tensor->place()),
                 tensor->data<void>(), platform::CPUPlace(), array,
                 tensor->numel() * sizeof(T), nullptr);
    delete[] array;
  }
}
};
};

