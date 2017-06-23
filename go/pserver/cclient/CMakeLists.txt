file(RELATIVE_PATH rel ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

go_extern(go_pserver ${rel}/...)
go_extern(go_logrus github.com/sirupsen/logrus)
go_library(paddle_pserver_cclient STATIC DEPS go_logrus go_pserver)

add_subdirectory(test)
