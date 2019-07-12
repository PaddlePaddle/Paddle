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
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
            if(NOT IOS_ENABLE_BITCODE)
                list(APPEND LIBS "-undefined dynamic_lookup")
            endif()
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
