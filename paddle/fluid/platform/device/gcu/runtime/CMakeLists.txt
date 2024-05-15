if(WITH_GCU)
  include_directories(/usr/include)
  include_directories(/usr/include/dtu)
  set(GCU_RUNTIME_SRC
      "gcu_rt_interface.cc"
      "rt_context.cc"
      "rt_event.cc"
      "rt_memory.cc"
      "rt_resources.cc"
      "rt_stream.cc"
      "rt_utils.cc"
      "rt_executable.cc")
  find_library(
    TOPSRT
    NAMES topsrt /usr/lib
    HINTS /opt/tops/lib)
  if(NOT TOPSRT)
    message(FATAL_ERROR "topsrt not found in system and /opt/tops/lib")
  endif()
  message("topsrt path:" ${TOPSRT})
  cc_library(
    gcu_runtime
    SRCS ${GCU_RUNTIME_SRC}
    DEPS dtu_sdk ${TOPSRT} efrt phi)
endif()
