#pragma once

#include "paddle/phi/core/enforce.h"
#include "cutlass/cutlass.h"


namespace phi{

namespace fusion {

namespace cutlass_internal {

#define PD_CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      PADDLE_THROW(phi::errors::Fatal("Got cutlass error: %s at: %d", cutlassGetStatusString(error), __LINE__)); \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


} // namespace phi

} // namespace fusion

} //namespace cutlass_internal
