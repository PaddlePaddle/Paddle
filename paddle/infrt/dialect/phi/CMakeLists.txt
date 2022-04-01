if (NOT INFRT_WITH_PHI)
    return()
endif()

add_subdirectory(ir)
add_subdirectory(pass)

add_executable(phi-exec phi_exec.cc)
target_link_libraries(phi-exec infrt)

gather_srcs(infrt_src SRCS
    data_type.cc)
