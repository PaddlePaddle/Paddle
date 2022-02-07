if (INFRT_WITH_PTEN)
  cc_library(pten_cpu_context SRCS pten_context.cc DEPS pten)
  cc_library(pten_cpu_allocator SRCS pten_allocator.cc DEPS pten)
endif()
