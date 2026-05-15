add_subdirectory(libuv)
add_subdirectory(googletest)

set(LIBTSP_BUILD_TESTS OFF)
add_subdirectory(libtsp)

if (NOT TARGET threadpark)
    add_subdirectory(threadpark)
endif ()
add_subdirectory(threadpool)

set(QUANT_BUILD_TESTS OFF)
set(QUANT_COMPILE_STATIC ON)
add_subdirectory(piquant)
