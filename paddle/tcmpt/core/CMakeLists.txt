IF(WITH_MKLDNN)
    set(MKLDNN_CTX_DEPS mkldnn)
ELSE()
    set(MKLDNN_CTX_DEPS)
ENDIF()

cc_library(backend SRCS backend.cc)
cc_library(dtype SRCS dtype.cc)
cc_library(layout SRCS layout.cc)

cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place backend dtype layout)
cc_library(dense_tensor SRCS dense_tensor.cc DEPS enforce data_type ddim allocator place convert_utils ${MKLDNN_CTX_DEPS})
cc_library(selected_rows_tensor SRCS selected_rows_tensor.cc DEPS dense_tensor)

cc_library(kernel_factory SRCS kernel_factory.cc DEPS enforce backend dtype layout)
cc_library(kernel_context SRCS kernel_context.cc DEPS enforce device_context)
