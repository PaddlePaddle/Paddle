# Some common routine for paddle compile.

# target_circle_link_libraries
# Link libraries to target which has circle dependencies.
#
# First Argument: target name want to be linked with libraries
# Rest Arguments: libraries which link together.
function(target_circle_link_libraries TARGET_NAME)
    if(APPLE)
        set(LIBS)
        set(inArchive OFF)
        set(libsInArgn)

        foreach(arg ${ARGN})
            if(${arg} STREQUAL "ARCHIVE_START")
                set(inArchive ON)
            elseif(${arg} STREQUAL "ARCHIVE_END")
                set(inArchive OFF)
            else()
                if(inArchive)
                    list(APPEND LIBS "-Wl,-force_load")
                endif()
                list(APPEND LIBS ${arg})
                list(APPEND libsInArgn ${arg})
            endif()
        endforeach()
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
            if(NOT IOS_ENABLE_BITCODE)
                list(APPEND LIBS "-undefined dynamic_lookup")
            endif()
        endif()
        list(REVERSE libsInArgn)
        target_link_libraries(${TARGET_NAME}
            ${LIBS}
            ${libsInArgn})

    else()  # LINUX
        set(LIBS)

        foreach(arg ${ARGN})
            if(${arg} STREQUAL "ARCHIVE_START")
                list(APPEND LIBS "-Wl,--whole-archive")
            elseif(${arg} STREQUAL "ARCHIVE_END")
                list(APPEND LIBS "-Wl,--no-whole-archive")
            else()
                list(APPEND LIBS ${arg})
            endif()
        endforeach()

        target_link_libraries(${TARGET_NAME}
                "-Wl,--start-group"
                ${LIBS}
                "-Wl,--end-group")
    endif()
endfunction()

# compile_cu_as_cpp
# Make a cu file compiled as C++
# Arguments: Source files
macro(compile_cu_as_cpp)
    foreach(s ${ARGN})
        set_source_files_properties(${s} PROPERTIES LANGUAGE CXX)
        set_source_files_properties(${s} PROPERTIES COMPILE_FLAGS "-x c++")
    endforeach()
endmacro()

# link_paddle_exe
# add paddle library for a paddle executable, such as trainer, pserver.
#
# It will handle WITH_PYTHON etc.
function(link_paddle_exe TARGET_NAME)
    if(WITH_RDMA)
        generate_rdma_links()
    endif()

    if(MOBILE_INFERENCE)
        target_circle_link_libraries(${TARGET_NAME}
            ARCHIVE_START
            paddle_gserver
            paddle_function
            ARCHIVE_END
            paddle_math
            paddle_utils
            paddle_parameter
            paddle_proto
            paddle_cuda
            ${EXTERNAL_LIBS}
            ${CMAKE_THREAD_LIBS_INIT}
            ${CMAKE_DL_LIBS}
            ${RDMA_LD_FLAGS}
            ${RDMA_LIBS})
    else()
        target_circle_link_libraries(${TARGET_NAME}
            ARCHIVE_START
            paddle_gserver
            paddle_function
            ARCHIVE_END
            paddle_pserver
            paddle_trainer_lib
            paddle_network
            paddle_math
            paddle_utils
            paddle_parameter
            paddle_proto
            paddle_cuda
            paddle_optimizer
            ${EXTERNAL_LIBS}
            ${CMAKE_THREAD_LIBS_INIT}
            ${CMAKE_DL_LIBS}
            ${RDMA_LD_FLAGS}
            ${RDMA_LIBS})
    endif()

    if(ANDROID)
        target_link_libraries(${TARGET_NAME} log)
    endif(ANDROID)

    if(WITH_MKLML AND MKLML_LIB_DIR AND MKLML_IOMP_LIB)
      target_link_libraries(${TARGET_NAME} "-L${MKLML_LIB_DIR} -liomp5 -Wl,--as-needed")
    endif()

    add_dependencies(${TARGET_NAME} ${external_project_dependencies})
endfunction()

# link_paddle_test
# Link a paddle unittest for target
# TARGET_NAME: the unittest target name
# Rest Arguemnts: not used.
function(link_paddle_test TARGET_NAME)
    link_paddle_exe(${TARGET_NAME})
    target_link_libraries(${TARGET_NAME}
                          paddle_test_main
                          paddle_test_util
                          ${GTEST_LIBRARIES})
endfunction()

# add_unittest_without_exec
#
# create a paddle unittest. not specifically define how to run this unittest.
# TARGET_NAME: the unittest target name, same as executable file name
# Rest Arguments: the source files to compile this unittest.
macro(add_unittest_without_exec TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    link_paddle_test(${TARGET_NAME})
endmacro()

# add_unittest
# create a paddle unittest and just to execute this binary to make unittest.
#
# TARGET_NAME: the unittest target name, same as executable file name
# Rest Arguments: the source files to compile this unittest.
macro(add_unittest TARGET_NAME)
    add_unittest_without_exec(${TARGET_NAME} ${ARGN})
    add_test(${TARGET_NAME} ${TARGET_NAME})
endmacro()

# add_simple_unittest
# create a paddle unittest with file name. It just compile ${TARGET_NAME}.cpp to
# ${TARGET_NAME} and then execute it.
macro(add_simple_unittest TARGET_NAME)
    add_unittest(${TARGET_NAME} ${TARGET_NAME}.cpp)
endmacro()

# Creates C resources file from files in given resource file
function(create_resources res_file output_file)
  add_custom_command(
    OUTPUT ${output_file}
    COMMAND python ARGS ${PADDLE_SOURCE_DIR}/cmake/make_resource.py ${res_file} ${output_file}
    DEPENDS ${res_file} ${PADDLE_SOURCE_DIR}/cmake/make_resource.py)
endfunction()
