file(GLOB_RECURSE ALL_TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

foreach(TEST_FILE ${ALL_TEST_FILES})
    get_filename_component (TARGET_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${TEST_FILE})
    add_dependencies(${TARGET_NAME} majel)
    target_link_libraries(${TARGET_NAME}     
                        ${GTEST_LIBRARIES}
                        ${GTEST_MAIN_LIBRARIES}
                        majel
                        )
    add_test(${TARGET_NAME} ${TARGET_NAME})
endforeach()
