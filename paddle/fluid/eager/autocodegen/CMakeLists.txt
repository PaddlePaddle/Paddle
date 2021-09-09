set(EAGER_GENERETOR_DEPS ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS} pybind proto_desc executor layer tracer engine imperative_profiler imperative_flag)

add_executable(eager_generator eager_generator.cc)
target_link_libraries(eager_generator ${EAGER_GENERETOR_DEPS})

get_property (os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
target_link_libraries(eager_generator ${os_dependency_modules})
