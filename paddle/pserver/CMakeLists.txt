# parameter server package

######################### paddle_network ####################
set(NETWORK_SOURCES
    LightNetwork.cpp
    SocketChannel.cpp
    ProtoServer.cpp)

set(NETWORK_HEADERS
    LightNetwork.h
    SocketChannel.h
    ProtoServer.h)

add_library(paddle_network STATIC
    ${NETWORK_SOURCES})

add_dependencies(paddle_network paddle_proto ${external_project_dependencies})

################### paddle_pserver ######################
set(PSERVER_SOURCES
    BaseClient.cpp
    ParameterClient2.cpp
    ParameterServer2.cpp
    SparseParameterDistribution.cpp
    ParameterServerController.cpp)

set(PSERVER_HEADERS
    BaseClient.h
    ParameterClient2.h
    ParameterServer2.h
    SparseParameterDistribution.h
    ParameterServerController.h)

add_library(paddle_pserver STATIC
    ${PSERVER_SOURCES})

add_dependencies(paddle_pserver paddle_proto ${external_project_dependencies})

set(PSERVER_MAIN_SOURCES
    ParameterServer2Main.cpp)

if(WITH_TESTING)
  add_subdirectory(test)
endif()

if(NOT MOBILE_INFERENCE)
  add_executable(paddle_pserver_main ${PSERVER_MAIN_SOURCES})
  link_paddle_exe(paddle_pserver_main)

  install(TARGETS paddle_pserver_main
          RUNTIME DESTINATION opt/paddle/bin
          PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
          GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)

  set_target_properties(paddle_pserver_main PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
