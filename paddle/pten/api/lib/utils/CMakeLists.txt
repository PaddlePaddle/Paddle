cc_library(pten_api_utils SRCS allocator.cc storage.cc tensor_utils.cc place_utils.cc DEPS
tensor_base convert_utils dense_tensor lod_tensor selected_rows place var_type_traits pten_common)
