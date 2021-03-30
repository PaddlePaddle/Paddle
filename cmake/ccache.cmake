# Use ccache if found ccache program

find_program(CCACHE_PATH ccache)

if(CCACHE_PATH)
    execute_process(COMMAND ccache -V OUTPUT_VARIABLE ccache_output)
    execute_process(COMMAND ccache -s cache directory OUTPUT_VARIABLE cache_directory)
    string(REGEX MATCH "[0-9]+.[0-9]+" ccache_version ${ccache_output})
    message(STATUS "Ccache is founded, use ccache to speed up compile.")
    # show statistics summary of ccache
    message("ccache version\t\t\t    " ${ccache_version} "\n" ${cache_directory})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PATH})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PATH})
endif(CCACHE_PATH)
