# Use ccache if found ccache program

find_program(CCACHE_FOUND ccache)

if(CCACHE_FOUND)
    message(STATUS "Ccache is founded, use ccache to speed up compile.")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)