if (WITH_DOUBLE)
  set(PADDLE_FLOAT_TYPE double)
else ()
  set(PADDLE_FLOAT_TYPE float)
endif()

# config.h used for C-API. It will store Paddle building configuration as a
# header. Make user just include PaddleCAPI.h then can get building
# configuration without explicitly set -DPADDLE_WITH_DOUBLE when building their
# libraries.
configure_file(config.h.in config.h @ONLY)

# PaddleCAPI.h is the only header we exposed. It currently only used for model
# inference.
file(GLOB CAPI_HEADERS *.h)
set(CAPI_PRIVATE_HEADER capi_private.h)
list(REMOVE_ITEM CAPI_HEADERS ${CAPI_PRIVATE_HEADER})
file(GLOB CAPI_SOURCES *.cpp)

# building paddle_capi
add_library(paddle_capi STATIC ${CAPI_HEADERS} ${CAPI_PRIVATE_HEADER}
  ${CAPI_SOURCES})

target_include_directories(paddle_capi PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_style_check_target(paddle_capi ${CAPI_SOURCES} ${CAPI_HEADER}
  ${CAPI_PRIVATE_HEADER})

add_dependencies(paddle_capi paddle_proto)


# combine all paddle static libraries together, into libpaddle_capi_whole.a
# user should use PaddleCAPI as -lpaddle_capi_whole
set(capi_whole_library libpaddle_capi_whole.a)
add_custom_target(paddle_capi_whole ALL
        COMMAND mkdir -p o_files/capi && cd o_files/capi/ && ar -x $<TARGET_FILE:paddle_capi>
        COMMAND mkdir -p o_files/utils && cd o_files/utils/ && ar -x $<TARGET_FILE:paddle_utils>
        COMMAND mkdir -p o_files/parameter && cd o_files/parameter/ && ar -x $<TARGET_FILE:paddle_parameter>
        COMMAND mkdir -p o_files/math && cd o_files/math/  && ar -x $<TARGET_FILE:paddle_math>
        COMMAND mkdir -p o_files/cuda && cd o_files/cuda/ && ar -x $<TARGET_FILE:paddle_cuda>
        COMMAND mkdir -p o_files/function && cd o_files/function/ && ar -x $<TARGET_FILE:paddle_function>
        COMMAND mkdir -p o_files/gserver && cd o_files/gserver/ && ar -x $<TARGET_FILE:paddle_gserver>
        COMMAND mkdir -p o_files/proto && cd o_files/proto/ && ar -x $<TARGET_FILE:paddle_proto>
        COMMAND mkdir -p o_files/network && cd o_files/network/ && ar -x $<TARGET_FILE:paddle_network>
        COMMAND mkdir -p o_files/pserver && cd o_files/pserver/ && ar -x $<TARGET_FILE:paddle_pserver>
        COMMAND ar crs ${capi_whole_library} `find ./o_files -name '*.o'`
        COMMAND rm -rf o_files
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS paddle_capi paddle_utils paddle_parameter paddle_math
                paddle_cuda paddle_function paddle_gserver
                paddle_proto paddle_pserver paddle_network
        )
set_target_properties(paddle_capi_whole
  PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${capi_whole_library})

set(LINK_FLAGS " -Wl,--retain-symbols-file ${CMAKE_CURRENT_SOURCE_DIR}/export.sym -Wl,--version-script ${CMAKE_CURRENT_SOURCE_DIR}/export.map")
# TODO: merge mkl into paddle_capi_shared
add_library(paddle_capi_shared SHARED ${CAPI_SOURCES})
set_target_properties(paddle_capi_shared	PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
target_include_directories(paddle_capi_shared PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
link_paddle_exe(paddle_capi_shared)

# install library & headers.
install(FILES ${CAPI_HEADERS} DESTINATION include/paddle)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h DESTINATION include/paddle)
if(ANDROID)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log --pretty=oneline -1
    OUTPUT_VARIABLE GIT_COMMITS_LIST
    RESULT_VARIABLE GIT_COMMITS_LIST_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(${GIT_COMMITS_LIST_RESULT})
    set(GIT_COMMITS_LIST "No commits.")
  endif()
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${capi_whole_library}
          DESTINATION lib/${ANDROID_ABI})
  install(TARGETS paddle_capi_shared DESTINATION lib/${ANDROID_ABI})
  install(CODE "FILE(WRITE ${CMAKE_INSTALL_PREFIX}/lib/${ANDROID_ABI}/BUILD.txt
          \"Compiler:\n\"
          \"\\t${CMAKE_C_COMPILER}\\n\"
          \"\\t${CMAKE_CXX_COMPILER}\\n\"
          \"Compiler Flags:\\n\"
          \"\\t${CMAKE_F_FLAGS}\\n\"
          \"\\t${CMAKE_CXX_FLAGS}\\n\"
          \"Android API: ${CMAKE_SYSTEM_VERSION}\\n\"
          \"Lastest commit:\\n\"
          \"\\t${GIT_COMMITS_LIST}\\n\"
      )"
  )
else(ANDROID)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${capi_whole_library} DESTINATION lib)
  install(TARGETS paddle_capi_shared DESTINATION lib)
endif(ANDROID)

# this variable used for unittest
set(PADDLE_CAPI_INC_PATH
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})

if (WITH_TESTING)
  add_subdirectory(tests)
endif()
