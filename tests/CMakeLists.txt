# When tests are built as a standalone project (not via add_subdirectory from
# the root), the pccl cmake target is not defined.  Use find_package so that
# the installed pcclConfig.cmake provides pccl::pccl with all of its
# transitive runtime dependencies (HIP, CUDA, …).
if (NOT TARGET pccl)
    find_package(pccl REQUIRED CONFIG)
    # Alias pccl::pccl → pccl so the individual test CMakeLists.txt work
    # without modification.
    add_library(pccl ALIAS pccl::pccl)
endif ()

add_subdirectory(basic_reduce_test)
add_subdirectory(concurrent_reduce_test)
add_subdirectory(c99_compat)
