set(PYTHON_TESTS_DIR
    ${PADDLE_BINARY_DIR}/python/paddle/base/tests
    CACHE INTERNAL "python tests directory")

add_subdirectory(utils)
add_subdirectory(common)
add_subdirectory(pir)
add_subdirectory(scripts)
add_subdirectory(testing)
add_subdirectory(phi)
add_subdirectory(fluid)

# NOTE(zhiqiu): The changes of cc tests
# Before, (1) the source file of cc tests are distributed in different sub-directories,
# (2) the tests are added and configured by calling `cc_test()` in each `CMakeLists.txt`,
# (3) the tests links static libraries of paddle modules,
# (4) the tests binaries are generated in different directories, as the same as the
# folder of source file.

# Now, we want to make all cc tests dynamically linked to the main paddle library,
# i.e., `libpaddle.so`, so we changes the logic of (2), (3), (4):
# (2) calling `cc_test()` in each `CMakeLists.txt` will not `exactly` add test, but
# record all tests and its source files, the action of add tests is deferred to HERE.
# Why doing so? since the target of `libpaddle.so` is mostly the last target, and
# the tests should be added after that according to dependency.
# (3) the tests links dynamic libraries, `libpaddle.so`
# (4) the tests are generated to the same directory, i.e., `CC_TESTS_DIR` defined above.

# Next, (to be discussed)
# (1) move all source files to same folder,
# (2) naturally, and configure tests in only one `CMakeLists.txt`,
# (3) cc tests support linking pre-built dynamic libraries. For example, use the dynamic
# library in the installed paddle by `pip`.
