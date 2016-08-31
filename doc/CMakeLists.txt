


if(NOT DEFINED SPHINX_THEME)
    set(SPHINX_THEME default)
endif()

if(NOT DEFINED SPHINX_THEME_DIR)
    set(SPHINX_THEME_DIR)
endif()

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")


set(PADDLE_DOXYGEN_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/doxygen_xml")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
    "${BINARY_BUILD_DIR}/conf.py"
    @ONLY)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    @ONLY
  )

add_custom_target(paddle_doxygen_docs ALL
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

sphinx_add_target(paddle_docs
                  html
                  ${BINARY_BUILD_DIR}
                  ${SPHINX_CACHE_DIR}
                  ${CMAKE_CURRENT_SOURCE_DIR}
                  ${SPHINX_HTML_DIR})

add_dependencies(paddle_docs 
  gen_proto_py
  paddle_doxygen_docs)