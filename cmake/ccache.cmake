# Use ccache if found ccache program

find_program(CCACHE_PATH ccache)

if(CCACHE_PATH)
    message(STATUS "Found ccache to accelerate compilation process.")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PATH})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PATH})
endif(CCACHE_PATH)
