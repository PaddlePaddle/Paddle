IF(WITH_MKLDNN)
    set(MKLDNN_CTX_DEPS mkldnn)
ELSE()
    set(MKLDNN_CTX_DEPS)
ENDIF()

cc_library(convert_utils SRCS convert_utils.cc DEPS data_type place)
cc_library(dense_tensor SRCS dense_tensor.cc DEPS enforce data_type ddim allocator place convert_utils ${MKLDNN_CTX_DEPS})
cc_library(selected_rows_tensor SRCS selected_rows.cc DEPS dense_tensor)
