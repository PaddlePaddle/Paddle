#pragma once

#ifdef PADDLE_WITH_XPU
#include "xpu/api.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"

namespace xpu           = baidu::xpu::api;
#endif
