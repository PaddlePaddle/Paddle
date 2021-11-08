add_subdirectory(utils)

# Adapt to custom op mechanism: Include the header files related to the data type
# to avoid exposing the path of the underlying file
include_directories(${PADDLE_SOURCE_DIR}/paddle/fluid/platform)

if (WITH_GPU)
  nv_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor tensor_meta pten_api_utils enforce)
elseif (WITH_ROCM)
  hip_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor tensor_meta pten_api_utils enforce)
else()
  cc_library(pten_tensor SRCS tensor.cc DEPS tensor_base dense_tensor tensor_meta pten_api_utils enforce)
endif()

cc_library(op_meta_info SRCS ext_op_meta_info.cc DEPS pten_tensor)

cc_library(math_api SRCS math.cc DEPS pten)
cc_library(linalg_api SRCS linalg.cc DEPS pten)
cc_library(creation_api SRCS creation.cc DEPS pten)
cc_library(manipulation_api SRCS manipulation.cc DEPS pten)
