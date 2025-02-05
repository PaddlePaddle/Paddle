if(WITH_CUSTOM_DEVICE AND NOT WIN32)
  cc_library(
    custom_device_load SRCS custom_device_load.cc custom_engine_load.cc
                            custom_engine_manager.cc)
endif()
