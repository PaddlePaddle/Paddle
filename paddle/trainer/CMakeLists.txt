# paddle trainer package

set(TRAINER_SOURCES
        ParameterUpdater.cpp
        ParamUtil.cpp
        RemoteParameterUpdater.cpp
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
        Tester.h
        TesterConfig.h
        Trainer.h
        TrainerInternal.h
        TrainerInternalConfig.h
        ThreadParameterUpdater.h
        TrainerConfigHelper.h)

add_library(paddle_trainer_lib STATIC
    ${TRAINER_SOURCES})

add_style_check_target(paddle_trainer_lib
    ${TRAINER_SOURCES})
add_style_check_target(paddle_trainer_lib
    ${TRAINER_HEADERS})
add_dependencies(paddle_trainer_lib
    gen_proto_cpp)

macro(add_paddle_exe TARGET_NAME)
  add_executable(${TARGET_NAME} ${ARGN})
  add_style_check_target(${TARGET_NAME} ${ARGN})
  link_paddle_exe(${TARGET_NAME})
endmacro()

add_paddle_exe(paddle_trainer
    TrainerMain.cpp)

add_paddle_exe(paddle_merge_model
    MergeModel.cpp)

if(WITH_TESTING)
    add_subdirectory(tests)
endif()
install(TARGETS paddle_trainer paddle_merge_model
    RUNTIME DESTINATION opt/paddle/bin
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)

set_target_properties(paddle_trainer PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(paddle_merge_model PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
