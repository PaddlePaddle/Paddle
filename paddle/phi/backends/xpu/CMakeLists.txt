cc_library(pten_xpu_info SRCS xpu_info.cc DEPS enforce xpulib pten_place)
cc_library(xpu_context SRCS xpu_context.cc DEPS pten_device_context pten_xpu_info)
