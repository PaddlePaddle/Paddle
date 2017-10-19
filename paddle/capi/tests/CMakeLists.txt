add_unittest(capi_test_mats test_Vector.cpp
  test_Matrix.cpp test_Arguments.cpp)

target_include_directories(capi_test_mats PUBLIC ${PADDLE_CAPI_INC_PATH})
target_link_libraries(capi_test_mats paddle_capi)

if(NOT MOBILE_INFERENCE)
    add_unittest_without_exec(capi_test_gradientMachine test_GradientMachine.cpp)
    target_include_directories(capi_test_gradientMachine PUBLIC
      ${PADDLE_CAPI_INC_PATH})
    target_link_libraries(capi_test_gradientMachine paddle_capi)
    add_test(NAME capi_test_gradientMachine
      COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python ${CMAKE_CURRENT_BINARY_DIR}/capi_test_gradientMachine
      WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle/capi/tests)
endif()
