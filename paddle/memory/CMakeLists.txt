add_subdirectory(detail)

if(${WITH_GPU})
  nv_library(memory SRCS memory.cc)
else(${WITH_GPU})
  cc_library(memory SRCS memroy.cc)
endif(${WITH_GPU})
