add_subdirectory(utils)

cc_library(ext_compat_utils SRCS ext_compat_utils.cc DEPS place)

if (WITH_GPU)
  nv_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor pten_api_utils ext_compat_utils enforce)
elseif (WITH_ROCM)
  hip_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor pten_api_utils ext_compat_utils enforce)
else()
  cc_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor pten_api_utils ext_compat_utils enforce)
endif()

cc_library(op_meta_info SRCS op_meta_info.cc DEPS pten_tensor)

cc_library(math_api SRCS math.cc DEPS pten)
cc_library(linalg_api SRCS linalg.cc DEPS pten)
cc_library(creation_api SRCS creation.cc DEPS pten)
cc_library(manipulation_api SRCS manipulation.cc DEPS pten)
