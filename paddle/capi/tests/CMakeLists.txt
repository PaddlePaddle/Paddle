function(add_capi_unittest TARGET_NAME)
  add_executable(
    ${TARGET_NAME}
    ${ARGN})
  target_link_libraries(
    ${TARGET_NAME}
    paddle_capi
    paddle_test_main
    ${GTEST_LIBRARIES})
  target_include_directories(${TARGET_NAME} PUBLIC ${PADDLE_CAPI_INC_PATH})
  add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endfunction()

add_capi_unittest(capi_test_mats test_Vector.cpp
  test_Matrix.cpp test_Arguments.cpp)
