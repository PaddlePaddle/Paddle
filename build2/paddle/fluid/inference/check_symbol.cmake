execute_process(COMMAND sh -c "/wangxianming/myPro/Paddle/paddle/fluid/inference/check_symbol.sh /wangxianming/myPro/Paddle/build2/paddle/fluid/inference/libpaddle_inference.so" RESULT_VARIABLE symbol_res)
if(NOT "${symbol_res}" STREQUAL "0")
  message(FATAL_ERROR "Check symbol failed.")
endif()
