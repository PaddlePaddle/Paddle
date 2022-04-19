add_subdirectory(collective)
add_subdirectory(store)
if(WITH_PYTHON)
  py_proto_compile(ps_py_proto SRCS the_one_ps.proto)
  add_custom_target(ps_py_proto_init ALL  
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PADDLE_BINARY_DIR}/python/paddle/distributed/fleet/proto)
  add_dependencies(ps_py_proto ps_py_proto_init)
  if (NOT WIN32)
    add_custom_command(TARGET ps_py_proto POST_BUILD
      COMMAND mv the_one_ps_pb2.py ${PADDLE_BINARY_DIR}/python/paddle/distributed/fleet/proto/)
  else(NOT WIN32)
    string(REPLACE "/" "\\" fleet_proto_dstpath "${PADDLE_BINARY_DIR}/python/paddle/distributed/fleet/proto/")
    add_custom_command(TARGET ps_py_proto POST_BUILD
      COMMAND copy /Y the_one_ps_pb2.py ${fleet_proto_dstpath}
      COMMENT "Copy generated python the_one_ps_pb2 into directory ${fleet_proto_dstpath}.")
  endif(NOT WIN32)
endif()

if(NOT WITH_PSCORE)
    add_subdirectory(fleet_executor)
    return()
endif()

proto_library(ps_framework_proto SRCS the_one_ps.proto)
add_custom_command(TARGET ps_framework_proto POST_BUILD
    COMMAND mv the_one_ps.pb.h ps.pb.h
    COMMAND mv the_one_ps.pb.cc ps.pb.cc)

set(DISTRIBUTE_COMPILE_FLAGS "-Wno-error=unused-value -Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=sign-compare -Wno-error=unused-variable -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=type-limits -Wno-error=unknown-pragmas -Wno-error=parentheses -Wno-error=unused-result")

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS
            "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

add_subdirectory(common)
add_subdirectory(ps)
add_subdirectory(test)
add_subdirectory(index_dataset)
add_subdirectory(fleet_executor)
