# paddle trainer package

set(TRAINER_SOURCES
        ParameterUpdater.cpp
        ParamUtil.cpp
        RemoteParameterUpdater.cpp
        NewRemoteParameterUpdater.cpp
        Tester.cpp
        Trainer.cpp
        TrainerInternal.cpp
        TrainerBenchmark.cpp
        ThreadParameterUpdater.cpp
        TrainerInternalConfig.cpp
        TrainerConfigHelper.cpp)

set(TRAINER_HEADERS
        ParameterUpdater.h
        ParamUtil.h
        RemoteParameterUpdater.h
        NewRemoteParameterUpdater.h
        Tester.h
        TesterConfig.h
        Trainer.h
        TrainerInternal.h
        TrainerInternalConfig.h
        ThreadParameterUpdater.h
        TrainerConfigHelper.h)

if(NOT WITH_GOLANG)
  list(REMOVE_ITEM TRAINER_SOURCES
          NewRemoteParameterUpdater.cpp)
  list(REMOVE_ITEM TRAINER_HEADERS
          NewRemoteParameterUpdater.h)
endif()

add_library(paddle_trainer_lib STATIC
    ${TRAINER_SOURCES})

add_dependencies(paddle_trainer_lib
    paddle_proto
    ${external_project_dependencies})

macro(add_paddle_exe TARGET_NAME)
  add_executable(${TARGET_NAME} ${ARGN})
  link_paddle_exe(${TARGET_NAME})
endmacro()

if(WITH_TESTING)
  add_subdirectory(tests)
endif()

if(NOT MOBILE_INFERENCE)
  add_paddle_exe(paddle_trainer TrainerMain.cpp)
  add_paddle_exe(paddle_merge_model MergeModel.cpp)

  install(TARGETS paddle_trainer paddle_merge_model
          RUNTIME DESTINATION opt/paddle/bin
          PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
          GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)

  set_target_properties(paddle_trainer PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
  set_target_properties(paddle_merge_model PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if(APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "-framework CoreFoundation -framework Security")
endif()

if(WITH_GOLANG)
  add_dependencies(paddle_trainer_lib paddle_pserver_cclient)
  target_link_libraries(paddle_trainer_lib paddle_pserver_cclient)
  target_link_libraries(paddle_trainer paddle_pserver_cclient)
endif(WITH_GOLANG)
