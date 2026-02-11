#pragma once

#include <memory>
#include <cstdint>

#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

class CustomDeviceFuncBase {
 public:
  virtual ~CustomDeviceFuncBase() = default;

  virtual void CustomCastDataType(const CustomContext& dev_ctx,
                                 const void* in,
                                 void* out,
                                 int64_t numel,
                                 DataType in_dtype,
                                 DataType out_dtype) const {
    PADDLE_THROW(common::errors::Unimplemented(
        "Custom Transform is not available in this build."));
  }
};

// Factory function declaration
std::unique_ptr<CustomDeviceFuncBase> CreateCustomDeviceFunc();

}  // namespace phi
