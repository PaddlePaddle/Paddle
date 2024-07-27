set(LOCAL_DETECTION_LIBS)

function(detection_library TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  set(options "")
  set(common_deps op_registry)
  set(pybind_flag 0)
  cmake_parse_arguments(detection_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  set(srcs)
  # filter cuda source file when not build with cuda/rocm
  foreach(src ${detection_library_SRCS})
    if(NOT WITH_GPU AND NOT WITH_ROCM)
      if(${src} MATCHES ".*\\.cc$")
        list(APPEND srcs ${src})
      endif()
    else()
      list(APPEND srcs ${src})
    endif()
  endforeach()

  op_library(${TARGET_NAME} SRCS ${srcs} DEPS ${common_deps}
             ${detection_library_DEPS})

  set(LOCAL_DETECTION_LIBS
      ${TARGET_NAME} ${LOCAL_DETECTION_LIBS}
      PARENT_SCOPE)
endfunction()

detection_library(anchor_generator_op SRCS anchor_generator_op.cc)

if(WITH_GPU OR WITH_ROCM)
  if(WITH_GPU)
    if(${CMAKE_CUDA_COMPILER_VERSION} LESS 11.0)
      set(TMPDEPS cub)
    endif()
  endif()
  detection_library(distribute_fpn_proposals_op SRCS
                    distribute_fpn_proposals_op.cc DEPS ${TMPDEPS})
else()
  detection_library(distribute_fpn_proposals_op SRCS
                    distribute_fpn_proposals_op.cc)
endif()

#Export local libraries to parent
# set(DETECTION_LIBRARY ${LOCAL_DETECTION_LIBS} PARENT_SCOPE)
