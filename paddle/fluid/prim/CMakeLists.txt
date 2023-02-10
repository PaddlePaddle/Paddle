add_subdirectory(api)
add_subdirectory(utils)
add_subdirectory(tests)
set(static_prim_deps prim_utils static_global_utils static_utils
                     static_prim_api)
set(eager_prim_deps prim_utils eager_prim_api)
