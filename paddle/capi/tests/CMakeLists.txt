add_unittest(capi_test_mats test_Vector.cpp
  test_Matrix.cpp test_Arguments.cpp)

target_include_directories(capi_test_mats PUBLIC ${PADDLE_CAPI_INC_PATH})
target_link_libraries(capi_test_mats paddle_capi)
add_unittest(capi_test_gradientMachine test_GradientMachine.cpp)

target_include_directories(capi_test_gradientMachine PUBLIC
  ${PADDLE_CAPI_INC_PATH})
target_link_libraries(capi_test_gradientMachine paddle_capi)
