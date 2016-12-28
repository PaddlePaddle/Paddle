configure_file(submit_local.sh.in
    submit_local.sh
    @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/submit_local.sh DESTINATION bin
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
            GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
        RENAME paddle)

configure_file(tools/usage_stat/usage.sh
    usage.sh
    @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/usage.sh DESTINATION opt/paddle/bin
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
            GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
        RENAME paddle_usage)
