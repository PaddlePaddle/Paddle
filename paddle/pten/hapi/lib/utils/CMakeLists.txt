add_subdirectory(tests)

cc_library(pten_hapi_utils SRCS allocator.cc storage.cc tensor_utils.cc DEPS tensor_base convert_utils
dense_tensor lod_tensor selected_rows place var_type_traits)
