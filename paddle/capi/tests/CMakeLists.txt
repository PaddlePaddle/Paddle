function(add_capi_unittest_without_exec TARGET_NAME)
  set(with_test_main ON)
  set(sources)
  foreach(source_file ${ARGN})
    if (${source_file} STREQUAL "NO_MAIN")
      set(with_test_main OFF)
    else()
      list(APPEND sources ${source_file})
    endif()
  endforeach()

  add_executable(
    ${TARGET_NAME}
    ${sources})


  target_link_libraries(
    ${TARGET_NAME}
    paddle_capi
    ${GTEST_LIBRARIES})

  if (with_test_main)
    target_link_libraries(
      ${TARGET_NAME} paddle_test_main)
  endif()
  target_include_directories(${TARGET_NAME} PUBLIC ${PADDLE_CAPI_INC_PATH})
endfunction()

function(add_capi_unittest TARGET_NAME)
  add_capi_unittest_without_exec(${TARGET_NAME} ${ARGN})
  add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endfunction()
add_capi_unittest(capi_test_mats test_Vector.cpp
  test_Matrix.cpp test_Arguments.cpp)

add_capi_unittest(capi_test_gradientMachine NO_MAIN test_GradientMachine.cpp)
