add_custom_target(paddle_apis ALL
                  DEPENDS paddle_v2_apis)

add_custom_target(paddle_docs ALL
                  DEPENDS paddle_v2_docs paddle_v2_docs_cn
                  paddle_mobile_docs paddle_mobile_docs_cn)

add_subdirectory(v2)
add_subdirectory(mobile)
