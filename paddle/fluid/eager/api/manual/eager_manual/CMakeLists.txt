add_subdirectory(forwards)
add_subdirectory(nodes)
set(eager_manual_functions
    ${eager_manual_functions}
    PARENT_SCOPE)
set(eager_manual_nodes
    ${eager_manual_nodes}
    PARENT_SCOPE)
