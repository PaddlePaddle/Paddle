file(GLOB_RECURSE ALL_TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp" "*.cc")

add_executable(majel_tests ${ALL_TEST_FILES})
add_dependencies(majel_tests majel ${external_project_dependencies})
target_link_libraries(majel_tests     
                      ${Boost_LIBRARIES}
                      ${GTEST_LIBRARIES}
                      majel
                     )
add_test(majel_tests majel_tests)
