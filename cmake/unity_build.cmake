set(UNITY_BEFORE_CODE [[
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif]])

function(register_unity_group TYPE)
    # Get UNITY_TARGET from CMAKE_CURRENT_SOURCE_DIR.
    string(REPLACE "${PADDLE_SOURCE_DIR}/paddle/fluid/" "" UNITY_TARGET ${CMAKE_CURRENT_SOURCE_DIR})
    string(REPLACE "/" "_" UNITY_TARGET ${UNITY_TARGET})
    set(UNITY_TARGET "paddle_${UNITY_TARGET}_unity")

    # Variable unity_group_index is used to record the number of UNITY_TARGET groups.
    get_property(unity_group_index GLOBAL PROPERTY ${UNITY_TARGET}_${TYPE}_group_index)
    if("${unity_group_index}" STREQUAL "")
        set(unity_group_index 0)
    endif()

    # Variable unity_group_sources is used to record the sources of one group.
    set(unity_group_sources ${UNITY_TARGET}_${TYPE}_group_${unity_group_index}_sources)
    set_property(GLOBAL PROPERTY ${unity_group_sources} "")
    foreach(src ${ARGN})
        # UB use absolute path of source.
        if(NOT IS_ABSOLUTE ${src})
            set(src ${CMAKE_CURRENT_SOURCE_DIR}/${src})
        endif()
        set_property(GLOBAL APPEND PROPERTY ${unity_group_sources} ${src})
    endforeach()

    set(unity_file ${CMAKE_CURRENT_BINARY_DIR}/${UNITY_TARGET}_${unity_group_index}.${TYPE})
    if(NOT EXISTS ${unity_file})
        file(TOUCH ${unity_file})
    endif()

    math(EXPR unity_group_index "${unity_group_index} + 1")
    set_property(GLOBAL PROPERTY ${UNITY_TARGET}_${TYPE}_group_index ${unity_group_index})
endfunction(register_unity_group)

function(compose_unity_target_sources TARGET TYPE)
    # Variable unity_target_sources represents the source file used in TARGET
    set(unity_target_sources "")
    get_property(unity_group_index_max GLOBAL PROPERTY ${TARGET}_${TYPE}_group_index)
    foreach(src ${ARGN})
        set(unity_file "")
        if(IS_ABSOLUTE ${src})
            set(src_absolute_path ${src})
        else()
            set(src_absolute_path ${CMAKE_CURRENT_SOURCE_DIR}/${src})
        endif()
        if(NOT "${unity_group_index_max}" STREQUAL "")
            math(EXPR unity_group_index_max "${unity_group_index_max} - 1")
            foreach(unity_group_index RANGE 0 ${unity_group_index_max})
                get_property(unity_group_sources GLOBAL PROPERTY ${TARGET}_${TYPE}_group_${unity_group_index}_sources)
                if(${src_absolute_path} IN_LIST unity_group_sources)
                    set(unity_file ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_${unity_group_index}.${TYPE})
                    set(unity_file_sources ${TARGET}_${TYPE}_file_${unity_group_index}_sources)
                    get_property(set_unity_file_sources GLOBAL PROPERTY ${unity_file_sources} SET)
                    if(NOT ${set_unity_file_sources})
                        set_property(GLOBAL PROPERTY ${unity_file_sources} "// Generate by Unity Build")
                        set_property(GLOBAL APPEND PROPERTY ${unity_file_sources} ${UNITY_BEFORE_CODE})
                    endif()
                    set_property(GLOBAL APPEND PROPERTY ${unity_file_sources} "#include \"${src_absolute_path}\"")
                    set(unity_target_sources ${unity_target_sources} ${unity_file})
                    break()
                endif()
            endforeach()
        endif()
        if("${unity_file}" STREQUAL "")
            set(unity_target_sources ${unity_target_sources} ${src})
        endif()
    endforeach()

    set(unity_target_sources ${unity_target_sources} PARENT_SCOPE)
endfunction(compose_unity_target_sources)

function(finish_unity_target TYPE)
    # Get UNITY_TARGET from CMAKE_CURRENT_SOURCE_DIR.
    string(REPLACE "${PADDLE_SOURCE_DIR}/paddle/fluid/" "" UNITY_TARGET ${CMAKE_CURRENT_SOURCE_DIR})
    string(REPLACE "/" "_" UNITY_TARGET ${UNITY_TARGET})
    set(UNITY_TARGET "paddle_${UNITY_TARGET}_unity")

    get_property(unity_group_index_max GLOBAL PROPERTY ${UNITY_TARGET}_${TYPE}_group_index)
    if(NOT "${unity_group_index_max}" STREQUAL "")
        math(EXPR unity_group_index_max "${unity_group_index_max} - 1")
        foreach(unity_group_index RANGE 0 ${unity_group_index_max})
            get_property(unity_file_sources GLOBAL PROPERTY ${UNITY_TARGET}_${TYPE}_file_${unity_group_index}_sources)
            set(unity_file_read_content "")
            string(JOIN "\n" unity_file_write_content ${unity_file_sources})
            set(unity_file ${CMAKE_CURRENT_BINARY_DIR}/${UNITY_TARGET}_${unity_group_index}.${TYPE})
            file(READ ${unity_file} unity_file_read_content)
            if(NOT "${unity_file_read_content}" STREQUAL "${unity_file_write_content}")
                file(WRITE ${unity_file} ${unity_file_write_content})
            endif()
        endforeach()
    endif()
endfunction(finish_unity_target)
