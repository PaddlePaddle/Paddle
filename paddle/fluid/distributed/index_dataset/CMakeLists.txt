proto_library(index_dataset_proto SRCS index_dataset.proto)
cc_library(
  index_wrapper
  SRCS index_wrapper.cc
  DEPS index_dataset_proto framework_io)
if(WITH_ONEDNN)
  cc_library(
    index_sampler
    SRCS index_sampler.cc
    DEPS xxhash index_wrapper eigen3 onednn)
else()
  cc_library(
    index_sampler
    SRCS index_sampler.cc
    DEPS xxhash index_wrapper eigen3)
endif()
if(WITH_PYTHON)
  py_proto_compile(index_dataset_py_proto SRCS index_dataset.proto)
endif()
