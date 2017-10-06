# util to check C++ file style
# * it basically use google cpplint.py.
# * It provide "add_style_check_target" for cmake.
#   Usage see add_style_check_target's document
#
# TODO(yuyang18): Add python style check.

set(STYLE_FILTER)

# diable unwanted filters

# paddle do not indent public/potected/private in class
set(STYLE_FILTER "${STYLE_FILTER}-whitespace/indent,")
# paddle use mutable reference. BUT IT IS NOT RECOMMANDED
set(STYLE_FILTER "${STYLE_FILTER}-runtime/references,")
# paddle use relative path for include.
set(STYLE_FILTER "${STYLE_FILTER}-build/include,")
# paddle use <thread>, <mutex>, etc.
set(STYLE_FILTER "${STYLE_FILTER}-build/c++11,")
# paddle use c style casting. BUT IT IS NOT RECOMMANDED
set(STYLE_FILTER "${STYLE_FILTER}-readability/casting")


# IGNORE SOME FILES
set(IGNORE_PATTERN
    .*ImportanceSampler.*
    .*cblas\\.h.*
    .*\\.pb\\.txt
    .*MultiDataProvider.*
    .*pb.*
    .*pybind.h)

# add_style_check_target
#
# attach check code style step for target.
#
# first argument: target name to attach
# rest arguments: source list to check code style.
#
# NOTE: If WITH_STYLE_CHECK is OFF, then this macro just do nothing.
macro(add_style_check_target TARGET_NAME)
    if(WITH_STYLE_CHECK)
        set(SOURCES_LIST ${ARGN})
        list(REMOVE_DUPLICATES SOURCES_LIST)
        foreach(filename ${SOURCES_LIST})
            foreach(pattern ${IGNORE_PATTERN})
                if(filename MATCHES ${pattern})
                    list(REMOVE_ITEM SOURCES_LIST ${filename})
                endif()
            endforeach()
        endforeach()

        if(SOURCES_LIST)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND "${PYTHON_EXECUTABLE}" "${PADDLE_SOURCE_DIR}/paddle/scripts/cpplint.py"
                        "--filter=${STYLE_FILTER}"
                        ${SOURCES_LIST}
                COMMENT "cpplint: Checking source code style"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})        
        endif()
    endif()
endmacro()
