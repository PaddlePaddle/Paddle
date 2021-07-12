cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place)
cc_library(base_tensor SRCS base_tensor.cc DEPS enforce data_type ddim allocator place convert_utils)
