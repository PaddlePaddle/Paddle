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
    .*LtrDataProvider.*
    .*MultiDataProvider.*)

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
        list(SORT SOURCES_LIST)

        foreach(filename ${SOURCES_LIST})
            set(LINT ON)
            foreach(pattern ${IGNORE_PATTERN})
                if(filename MATCHES ${pattern})
                    message(STATUS "DROP LINT ${filename}")
                    set(LINT OFF)
                endif()
            endforeach()
            if(LINT MATCHES ON)
                get_filename_component(base_filename ${filename} NAME)
                set(CUR_GEN ${CMAKE_CURRENT_BINARY_DIR}/${base_filename}.cpplint)
                add_custom_command(OUTPUT ${CUR_GEN}
                    PRE_BUILD
                    COMMAND env ${py_env} "${PYTHON_EXECUTABLE}" "${PROJ_ROOT}/paddle/scripts/cpplint.py"
                                "--filter=${STYLE_FILTER}"
                                "--write-success=${CUR_GEN}" ${filename}
                    DEPENDS ${filename}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
            endif()
        endforeach()
    endif()
endmacro()
