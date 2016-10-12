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
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            list(APPEND LIBS "-undefined dynamic_lookup")
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
# It will handle WITH_PYTHON/WITH_GLOG etc.
function(link_paddle_exe TARGET_NAME)
    if(WITH_METRIC)
        if(WITH_GPU)
            set(METRIC_LIBS paddle_metric_learning paddle_dserver_lib metric metric_cpu)
        else()
            set(METRIC_LIBS paddle_metric_learning paddle_dserver_lib metric_cpu)
        endif()
    else()
        set(METRIC_LIBS "")
    endif()

    if(PADDLE_WITH_INTERNAL)
        set(INTERAL_LIBS paddle_internal_gserver paddle_internal_parameter)
        target_circle_link_libraries(${TARGET_NAME}
            ARCHIVE_START
            paddle_internal_gserver
            paddle_internal_owlqn
            ARCHIVE_END
            paddle_internal_parameter)
    else()
        set(INTERAL_LIBS "")
    endif()

    target_circle_link_libraries(${TARGET_NAME}
        ARCHIVE_START
        paddle_gserver
        ${METRIC_LIBS}
        ARCHIVE_END
        paddle_pserver
        paddle_trainer_lib
        paddle_network
        paddle_math
        paddle_utils
        paddle_parameter
        paddle_proto
        paddle_cuda
        ${METRIC_LIBS}
        ${PROTOBUF_LIBRARY}
        ${CMAKE_THREAD_LIBS_INIT}
        ${CBLAS_LIBS}
        ${ZLIB_LIBRARIES}
        ${INTERAL_LIBS}
        ${CMAKE_DL_LIBS})
    
    if(WITH_PYTHON)
        target_link_libraries(${TARGET_NAME}
            ${PYTHON_LIBRARIES})
    endif()

    if(WITH_GLOG)
        target_link_libraries(${TARGET_NAME}
            ${LIBGLOG_LIBRARY})
    endif()

    if(WITH_GFLAGS)
        target_link_libraries(${TARGET_NAME}
            ${GFLAGS_LIBRARIES})
    endif()

    if(WITH_GPU)
        if(NOT WITH_DSO OR WITH_METRIC) 
            target_link_libraries(${TARGET_NAME}
                ${CUDNN_LIBRARY}
                ${CUDA_curand_LIBRARY}) 
            CUDA_ADD_CUBLAS_TO_TARGET(${TARGET_NAME})
        endif()

        check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME )
        if(HAVE_CLOCK_GETTIME)
            target_link_libraries(${TARGET_NAME} rt)
        endif()
    endif()
endfunction()

# link_paddle_test
# Link a paddle unittest for target
# TARGET_NAME: the unittest target name
# Rest Arguemnts: not used.
function(link_paddle_test TARGET_NAME)
    link_paddle_exe(${TARGET_NAME})
    target_link_libraries(${TARGET_NAME} ${GTEST_MAIN_LIBRARIES}
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
    add_style_check_target(${TARGET_NAME} ${ARGN})
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
function(create_resources res_file output)
    # Create empty output file
    file(WRITE ${output} "")
    # Get short filename
    string(REGEX MATCH "([^/]+)$" filename ${res_file})
    # Replace filename spaces & extension separator for C compatibility
    string(REGEX REPLACE "\\.| |-" "_" filename ${filename})
    # Read hex data from file
    file(READ ${res_file} filedata HEX)
    # Convert hex data for C compatibility
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})
    # Append data to output file
    file(APPEND ${output} "const unsigned char ${filename}[] = {${filedata}};\nconst unsigned ${filename}_size = sizeof(${filename});\n")
endfunction()
