if(NOT DEFINED SPHINX_THEME)
    set(SPHINX_THEME default)
endif()

if(NOT DEFINED SPHINX_THEME_DIR)
    set(SPHINX_THEME_DIR)
endif()

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR_EN "${CMAKE_CURRENT_BINARY_DIR}/en/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR_EN "${CMAKE_CURRENT_BINARY_DIR}/en/_doctrees")

# HTML output director
set(SPHINX_HTML_DIR_EN "${CMAKE_CURRENT_BINARY_DIR}/en/html")

set(IMPORT_PADDLE_STRING "")
set(IMPORT_PADDLEV2_STRING "")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../templates/conf.py.en.in"
    "${BINARY_BUILD_DIR_EN}/conf.py"
    @ONLY)

sphinx_add_target(paddle_v2_docs
                  html
                  ${BINARY_BUILD_DIR_EN}
                  ${SPHINX_CACHE_DIR_EN}
                  ${CMAKE_CURRENT_SOURCE_DIR}
                  ${SPHINX_HTML_DIR_EN})

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR_CN "${CMAKE_CURRENT_BINARY_DIR}/cn/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR_CN "${CMAKE_CURRENT_BINARY_DIR}/cn/_doctrees")

# HTML output directory
set(SPHINX_HTML_DIR_CN "${CMAKE_CURRENT_BINARY_DIR}/cn/html")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../templates/conf.py.cn.in"
    "${BINARY_BUILD_DIR_CN}/conf.py"
    @ONLY)

sphinx_add_target(paddle_v2_docs_cn
                  html
                  ${BINARY_BUILD_DIR_CN}
                  ${SPHINX_CACHE_DIR_CN}
                  ${CMAKE_CURRENT_SOURCE_DIR}
                  ${SPHINX_HTML_DIR_CN})

add_subdirectory(api)
