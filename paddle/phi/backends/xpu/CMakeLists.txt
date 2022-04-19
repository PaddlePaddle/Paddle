cc_library(phi_xpu_info SRCS xpu_info.cc DEPS enforce xpulib phi_place)
cc_library(xpu_context SRCS xpu_context.cc DEPS phi_device_context phi_xpu_info)
