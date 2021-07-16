cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place)
cc_library(base_tensor SRCS base_tensor.cc DEPS enforce data_type ddim allocator place convert_utils)
cc_library(pten_lod_tensor SRCS lod_tensor.cc DEPS base_tensor)
cc_library(pten_selected_rows SRCS lod_tensor.cc DEPS base_tensor)
