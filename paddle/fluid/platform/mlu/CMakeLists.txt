IF(WITH_MLU)
    cc_binary(mlu_info SRCS mlu_info.cc DEPS glog neuware_lib)
ENDIF()