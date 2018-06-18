if (NOT WITH_ANAKIN)
  return()
endif()

set(ANAKIN_INSTALL_DIR "${THIRD_PARTY_PATH}/install/anakin" CACHE PATH
  "Anakin install path." FORCE)
set(ANAKIN_INCLUDE "${ANAKIN_INSTALL_DIR}" CACHE STRING "root of Anakin header files")
set(ANAKIN_LIBRARY "${ANAKIN_INSTALL_DIR}" CACHE STRING "path of Anakin library")

set(ANAKIN_COMPILE_EXTRA_FLAGS -Wno-error=unused-variable -Wno-error=format-extra-args -Wno-error=comment -Wno-error=format -Wno-error=switch -Wno-error=return-type -Wno-error=non-virtual-dtor -Wno-reorder -Wno-error=cpp)

set(ANAKIN_LIBRARY_URL "https://github.com/pangge/Anakin/releases/download/3.0/anakin_release_simple.tar.gz")

# A helper function used in Anakin, currently, to use it, one need to recursively include
# nearly all the header files.
function(fetch_include_recursively root_dir)
    if (IS_DIRECTORY ${root_dir})
        include_directories(${root_dir})
    endif()

    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*)
    foreach(sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub})
            fetch_include_recursively(${root_dir}/${sub})
        endif()
    endforeach()
endfunction()

# download library
message(STATUS "Download Anakin library from ${ANAKIN_LIBRARY_URL}")
execute_process(COMMAND bash -c "mkdir -p ${ANAKIN_INSTALL_DIR}")
execute_process(COMMAND bash -c "rm -rf ${ANAKIN_INSTALL_DIR}/*")
execute_process(COMMAND bash -c "cd ${ANAKIN_INSTALL_DIR}; wget -q ${ANAKIN_LIBRARY_URL}")
execute_process(COMMAND bash -c "mkdir -p ${ANAKIN_INSTALL_DIR}")
execute_process(COMMAND bash -c "cd ${ANAKIN_INSTALL_DIR}; tar xzf anakin_release_simple.tar.gz")

if (WITH_ANAKIN)
    message(STATUS "Anakin for inference is enabled")
    message(STATUS "Anakin is set INCLUDE:${ANAKIN_INCLUDE} LIBRARY:${ANAKIN_LIBRARY}")
    fetch_include_recursively(${ANAKIN_INCLUDE})
    link_directories(${ANAKIN_LIBRARY})
endif()
