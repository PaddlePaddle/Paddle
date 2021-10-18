include(tcmpt)
# tcmpt api
add_subdirectory(api)
# tcmpt high level api
add_subdirectory(hapi)
# tcmpt core components
add_subdirectory(core)
# tcmpt kernels for diff device
add_subdirectory(kernels)
# tcmpt infershape
add_subdirectory(infershape)
# TODO(xingfeng): tcmpt inner module API designed by a high-performance team
add_subdirectory(module)
# tcmpt tests
add_subdirectory(tests)
