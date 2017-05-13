file(GLOB_RECURSE ALL_TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

add_executable(majel_tests ${ALL_TEST_FILES})
add_dependencies(majel_tests majel)
target_link_libraries(majel_tests     
                      ${Boost_LIBRARIES}
                      ${GTEST_LIBRARIES}
                      ${GTEST_MAIN_LIBRARIES}
                      majel
                     )
add_test(majel_tests majel_tests)
