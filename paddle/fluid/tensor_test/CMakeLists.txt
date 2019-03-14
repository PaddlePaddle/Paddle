set(PYBIND_DEPS pybind python proto_desc memory )

if(WITH_PYTHON)
    list(APPEND PYBIND_DEPS py_func_op)
endif()
set(PYBIND_SRCS test.cc exception.cc)

if(WITH_PYTHON)
    if(WITH_AMD_GPU)
        hip_library(tensor_test_ex SHARED
                SRCS ${PYBIND_SRCS}
                DEPS ARCHIVE_START ${PYBIND_DEPS}
               ARCHIVE_END)
    else()
        cc_library(tensor_test_ex SHARED
                SRCS ${PYBIND_SRCS}
                DEPS ${PYBIND_DEPS}
                 )
        if(NOT APPLE AND NOT WIN32)
            target_link_libraries(tensor_test_ex rt)
        endif(NOT APPLE AND NOT WIN32)
    endif(WITH_AMD_GPU)

    get_property (os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
    target_link_libraries(tensor_test_ex ${os_dependency_modules})

#    cc_test(tensor_py_test SRCS tensor_py_test.cc DEPS python pybind)

    IF(APPLE)
        add_custom_command(TARGET tensor_test_ex POST_BUILD
            COMMAND cmake -E copy /Users/peizhilin/Desktop/me/mac_build/paddle/fluid/tensor_test/libtensor_test.dylib
            /Users/peizhilin/Desktop/me/mac_build/paddle/fluid/tensor_test/tensor_test.so
            )
    else()
        add_custom_command(TARGET tensor_test_ex POST_BUILD
                COMMAND cmake -E copy /libtensor_test.dll
                c:/Users/peizhilin/Desktop/tensor_test.dll
                )
    endif()
endif(WITH_PYTHON)
