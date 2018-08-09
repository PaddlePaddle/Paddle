if (NOT WITH_ANAKIN)
  return()
endif()

set(ANAKIN_SOURCE_DIR  ${THIRD_PARTY_PATH}/anakin)
# the anakin install dir is only default one now
set(ANAKIN_INSTALL_DIR ${THIRD_PARTY_PATH}/anakin/src/extern_anakin/output)
set(ANAKIN_INCLUDE     ${ANAKIN_INSTALL_DIR})
set(ANAKIN_LIBRARY     ${ANAKIN_INSTALL_DIR})
SET(ANAKIN_SHARED_LIB  ${ANAKIN_LIBRARY}/libanakin.so)
SET(ANAKIN_SABER_LIB   ${ANAKIN_LIBRARY}/libanakin_saber_common.so)

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
fetch_include_recursively(${ANAKIN_INCLUDE})

# A nother helper function used in Anakin.
function(target_fetch_include_recursively root_dir target_name)
    if (IS_DIRECTORY ${root_dir})
        target_include_directories(${target_name} PUBLIC ${root_dir})
    endif()

    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*)
    foreach(sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub})
            target_include_directories(${target_name} PUBLIC ${root_dir}/${sub})
        endif()
    endforeach()
endfunction()

set(ANAKIN_COMPILE_EXTRA_FLAGS 
    -Wno-error=unused-but-set-variable -Wno-unused-but-set-variable
    -Wno-error=unused-variable -Wno-unused-variable 
    -Wno-error=format-extra-args -Wno-format-extra-args
    -Wno-error=comment -Wno-comment 
    -Wno-error=format -Wno-format 
    -Wno-error=switch -Wno-switch
    -Wno-error=return-type -Wno-return-type 
    -Wno-error=non-virtual-dtor -Wno-non-virtual-dtor
    -Wno-sign-compare
    -Wno-reorder 
    -Wno-error=cpp)

ExternalProject_Add(
    extern_anakin
    ${EXTERNAL_PROJECT_LOG_ARGS}
    # TODO(luotao): use PaddlePaddle/Anakin later
    GIT_REPOSITORY      "https://github.com/luotao1/Anakin"
    GIT_TAG             "3957ae9263eaa0b1986758dac60a88852afb09be"
    PREFIX              ${ANAKIN_SOURCE_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DUSE_GPU_PLACE=YES
                        -DUSE_X86_PLACE=YES
                        -DBUILD_WITH_UNIT_TEST=NO
                        -DPROTOBUF_ROOT=${THIRD_PARTY_PATH}/install/protobuf
                        -DMKLML_ROOT=${THIRD_PARTY_PATH}/install/mklml
                        -DCUDNN_ROOT=${CUDNN_ROOT}
                        ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS    -DCMAKE_INSTALL_PREFIX:PATH=${ANAKIN_INSTALL_DIR}
)

message(STATUS "Anakin for inference is enabled")
message(STATUS "Anakin is set INCLUDE:${ANAKIN_INCLUDE} LIBRARY:${ANAKIN_LIBRARY}")

add_dependencies(extern_anakin protobuf mklml)
add_library(anakin SHARED IMPORTED GLOBAL)
set_property(TARGET anakin PROPERTY IMPORTED_LOCATION ${ANAKIN_SHARED_LIB})
set_property(TARGET anakin PROPERTY IMPORTED_LOCATION ${ANAKIN_SABER_LIB})
set_property(TARGET anakin PROPERTY IMPORTED_LOCATION ${CUDNN_LIBRARY})
add_dependencies(anakin extern_anakin)
list(APPEND external_project_dependencies anakin)
