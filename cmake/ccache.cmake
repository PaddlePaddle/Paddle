# Use ccache if found ccache program

if(NOT WIN32)
    find_program(CCACHE_PATH ccache)
    if(CCACHE_PATH)
        execute_process(COMMAND ccache -V OUTPUT_VARIABLE ccache_output)
        execute_process(COMMAND ccache -s cache directory OUTPUT_VARIABLE cache_directory)
        string(REGEX MATCH "[0-9]+.[0-9]+" ccache_version ${ccache_output})
        message(STATUS "ccache is founded, use ccache to speed up compile on Unix.")
        # show statistics summary of ccache
        message("ccache version\t\t\t    " ${ccache_version} "\n" ${cache_directory})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PATH})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PATH})
    endif(CCACHE_PATH)
elseif("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    # (Note:zhouwei25) Only Ninja Generator can support sccache now
    find_program(SCCACHE_PATH sccache)

    if(SCCACHE_PATH)
        execute_process(COMMAND sccache -V OUTPUT_VARIABLE sccache_version)
        message(STATUS "sccache is founded, use [${SCCACHE_PATH}] to speed up compile on Windows.")

        set(CMAKE_C_COMPILER_LAUNCHER ${SCCACHE_PATH})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_PATH})
        # (Note:zhouwei25) sccache for cuda compiler has bug so that it can't be hit
        # refer to https://github.com/mozilla/sccache/issues/1017, so we fix it
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${SCCACHE_PATH})
    endif(SCCACHE_PATH)
endif()
