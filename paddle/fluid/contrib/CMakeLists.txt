if(WITH_WBAES)
  cc_library(cryption SRCS cryption.cc DEPS wbaes)
  cc_test(cryption_test SRCS cryption_test.cc DEPS cryption glog)
endif()
