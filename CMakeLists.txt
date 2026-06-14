cmake_minimum_required(VERSION 3.16)
project(pccl)

include(cmake/macos-openmp.cmake)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# make sure we reference dynamic libraries such as OpenMP via RPATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_MACOSX_RPATH ON)

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@rpath")
elseif(UNIX)
    # For Linux, we often use $ORIGIN so it’s relative to the .so location.
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

# Detect CPU architecture
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(IS_AMD64 TRUE)
else ()
    set(IS_AMD64 FALSE)
endif ()
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(IS_ARM64 TRUE)
else ()
    set(IS_ARM64 FALSE)
endif ()

# Handle CUDA build
option(PCCL_BUILD_CUDA_SUPPORT "Build PCCL with CUDA support" ON)
option(PCCL_BUILD_OPENMP_SUPPORT "Build PCCL with OpenMP support" ON)
option(PCCL_BUILD_TESTS "Build PCCL tests" ON)

if (PCCL_BUILD_CUDA_SUPPORT)
    find_package(CUDAToolkit)

    # check if the cuda toolkit was found
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit was found!")
        # You can now use CUDA-specific targets and include directories.
    else()
        message(WARNING "CUDA Toolkit not found! CUDA support will be disabled!")
        set(PCCL_BUILD_CUDA_SUPPORT OFF)
    endif()
endif ()

if (PCCL_BUILD_OPENMP_SUPPORT)
    set(OpenMP_USE_STATIC_LIBS ON)  # Request static OpenMP
    find_package(OpenMP QUIET)

    if (OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP. Compiling with OpenMP support")
        set(PCCL_BUILD_OPENMP_SUPPORT ON)
    else()
        message(STATUS "OpenMP not found through find_package(OpenMP). Checking compiler flags manually...")

        # Manual check of compiler flags
        include(CheckCXXCompilerFlag)

        if (MSVC)
            # MSVC uses /openmp
            check_cxx_compiler_flag("/openmp" MSVC_OPENMP_SUPPORTED)
            if (MSVC_OPENMP_SUPPORTED)
                message(STATUS "OpenMP supported by MSVC. Enabling /openmp flag.")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
            else()
                message(STATUS "OpenMP is not supported by this version of MSVC.")
                set(PCCL_BUILD_OPENMP_SUPPORT OFF)
            endif()
        else()
            # GCC, Clang, etc. typically use -fopenmp
            check_cxx_compiler_flag("-fopenmp" GCC_CLANG_OPENMP_SUPPORTED)
            if (GCC_CLANG_OPENMP_SUPPORTED)
                message(STATUS "OpenMP supported by GCC/Clang. Enabling -fopenmp flag.")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
            else()
                message(STATUS "OpenMP is not supported by this compiler or version.")
                set(PCCL_BUILD_OPENMP_SUPPORT OFF)
            endif()
        endif()
    endif()
else()
    message(STATUS "PCCL_BUILD_OPENMP_SUPPORT is OFF. Skipping OpenMP checks.")
endif()

if (PCCL_BUILD_OPENMP_SUPPORT)
    add_compile_definitions(PCCL_BUILD_OPENMP_SUPPORT)
endif ()

if (PCCL_BUILD_CUDA_SUPPORT)
    set(CMAKE_CUDA_STANDARD 20)

    if (CMAKE_CUDA_ARCHITECTURES)
        set(PCCL_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    else ()
        set(PCCL_CUDA_ARCHITECTURES "50;52;53;60;61;62;70;72;75;80;86;87;89;90;90a")
    endif ()

    # enable cuda language
    enable_language(CUDA)
endif ()

# Handle HIP build
option(PCCL_BUILD_HIP_SUPPORT "Build PCCL with HIP/ROCm support" OFF)

if (PCCL_BUILD_HIP_SUPPORT)
    if (CMAKE_VERSION VERSION_LESS 3.21)
        message(WARNING "HIP language support requires CMake >= 3.21 (found ${CMAKE_VERSION}). HIP support disabled.")
        set(PCCL_BUILD_HIP_SUPPORT OFF)
    else()
        # Prefer the versioned ROCm installation over any system-installed HIP package,
        # which may reference renamed or incompatible library files.
        if (NOT hip_DIR AND NOT HIP_DIR)
            find_path(_rocm_root lib/cmake/hip/hip-config.cmake
                PATHS /opt/rocm-7.2.4 /opt/rocm NO_DEFAULT_PATH)
            if (_rocm_root)
                list(PREPEND CMAKE_PREFIX_PATH "${_rocm_root}")
            endif ()
        endif ()
        find_package(hip QUIET)
        if (hip_FOUND)
            message(STATUS "HIP/ROCm was found!")
            set(CMAKE_HIP_STANDARD 17)
            if (CMAKE_HIP_ARCHITECTURES)
                set(PCCL_HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
            else ()
                set(PCCL_HIP_ARCHITECTURES "gfx1100;gfx1101;gfx1200;gfx1201")
            endif ()
            enable_language(HIP)
        else()
            message(WARNING "HIP/ROCm not found! HIP support will be disabled!")
            set(PCCL_BUILD_HIP_SUPPORT OFF)
        endif()
    endif()
endif ()

option(PCCL_BUILD_STATIC_LIB "Build PCCL as a static library" ON)

# When building a static pccl, force libuv to also build as static so its
# symbols can be merged into the fat libpccl.a archive.
if (PCCL_BUILD_STATIC_LIB)
    set(LIBUV_BUILD_SHARED OFF CACHE BOOL "Build libuv as shared" FORCE)
endif ()

# add third party libraries
add_subdirectory(third_party)

if (PCCL_BUILD_TESTS)
    enable_testing()

    # include testing functions
    include(cmake/testing.cmake)
endif ()

# add internal libraries
add_subdirectory(common_utils)
add_subdirectory(log)
add_subdirectory(ccoip)
add_subdirectory(ccoip_master)
add_subdirectory(tinysockets)

if (PCCL_BUILD_STATIC_LIB)
    add_library(pccl STATIC src/pccl.cpp)
else ()
    add_library(pccl SHARED src/pccl.cpp)
endif ()

if (NOT WIN32)
    target_compile_options(pccl PRIVATE -Wall -Wextra -Werror -Wpedantic -Wno-unused-parameter)
endif ()

if (PCCL_BUILD_CUDA_SUPPORT)
    target_compile_definitions(pccl PRIVATE PCCL_HAS_CUDA_SUPPORT)
endif ()

if (PCCL_BUILD_HIP_SUPPORT)
    target_compile_definitions(pccl PRIVATE PCCL_HAS_HIP_SUPPORT)
endif ()

target_include_directories(pccl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(pccl PRIVATE internal)

target_link_libraries(pccl PRIVATE $<BUILD_INTERFACE:ccoip_private>)
target_link_libraries(pccl PRIVATE $<BUILD_INTERFACE:ccoip_public>)
target_link_libraries(pccl PRIVATE $<BUILD_INTERFACE:pccl_log>)
# Expose the ccoip public headers through pccl's own include interface so
# they are available both in the build tree and via the installed cmake target.
target_include_directories(pccl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ccoip/public_include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# When pccl is a static library, consumers must link its runtime deps themselves
# because archive files don't carry shared-library transitive requirements.
# Use plain linker flags (not cmake target names) so the installed cmake
# package config works without requiring consumers to call find_package(HIP).
if (PCCL_BUILD_STATIC_LIB)
    if (PCCL_BUILD_HIP_SUPPORT AND hip_FOUND)
        # hip::amdhip64 only sets IMPORTED_LOCATION_RELWITHDEBINFO, not the
        # generic IMPORTED_LOCATION.  Use find_library to reliably locate the
        # shared library so we can emit a proper -L<dir> -lamdhip64 pair.
        find_library(_amdhip64_lib NAMES amdhip64
            HINTS "${HIP_PATH}/lib" /opt/rocm/lib /opt/rocm-7.2.4/lib
            NO_DEFAULT_PATH)
        if (_amdhip64_lib)
            get_filename_component(_hip_lib_dir "${_amdhip64_lib}" DIRECTORY)
            target_link_libraries(pccl INTERFACE "-L${_hip_lib_dir}" -lamdhip64)
        else ()
            target_link_libraries(pccl INTERFACE -lamdhip64)
        endif ()
    endif ()
    if (PCCL_BUILD_CUDA_SUPPORT AND CUDAToolkit_FOUND)
        target_link_libraries(pccl INTERFACE CUDA::cuda_driver)
    endif ()
        if (PCCL_BUILD_OPENMP_SUPPORT AND OpenMP_CXX_FOUND)
            # Use the resolved library paths (not the cmake target name) so
            # the exported cmake package works without requiring consumers to
            # call find_package(OpenMP) themselves.
            target_link_libraries(pccl INTERFACE ${OpenMP_CXX_LIBRARIES})
        endif ()
    fix_macos_abspaths(pccl)
endif ()

# ---------------------------------------------------------------------------
# Fat static library: merge all component archives into a single libpccl.a so
# that external users can link with just -lpccl without needing to supply
# every transitive dependency on the link line.
# ---------------------------------------------------------------------------
if (PCCL_BUILD_STATIC_LIB AND UNIX)
    # Base set of component archives (platform-independent)
    set(_pccl_fat_libs
        $<TARGET_FILE:pccl_log>
        $<TARGET_FILE:ccoip_internal>
        $<TARGET_FILE:crc32_cpu>
        $<TARGET_FILE:simplehash_cpu>
        $<TARGET_FILE:tinysockets>
        $<TARGET_FILE:tsp>
        $<TARGET_FILE:threadpool>
        $<TARGET_FILE:threadpark>
        $<TARGET_FILE:piquant>
        $<TARGET_FILE:uv_a>
    )
    if (IS_AMD64)
        list(APPEND _pccl_fat_libs
            $<TARGET_FILE:crc32_amd64_sse42>
            $<TARGET_FILE:crc32_amd64_sse42_pcmul>
        )
    elseif (IS_ARM64)
        list(APPEND _pccl_fat_libs $<TARGET_FILE:crc32_arm64>)
    endif ()
    if (PCCL_BUILD_CUDA_SUPPORT AND CUDAToolkit_FOUND)
        list(APPEND _pccl_fat_libs $<TARGET_FILE:simplehash_cuda>)
    endif ()

    # Build the ar MRI script content at generation time so generator
    # expressions (TARGET_FILE) are resolved to real paths.
    set(_mri_addlibs "")
    foreach(_f IN LISTS _pccl_fat_libs)
        string(APPEND _mri_addlibs "ADDLIB ${_f}\n")
    endforeach()

    if (NOT APPLE)
        # Linux: ar -M with an MRI script
        set(_mri_file "${CMAKE_CURRENT_BINARY_DIR}/pccl_fat.mri")
        file(GENERATE OUTPUT "${_mri_file}"
            CONTENT "CREATE $<TARGET_FILE:pccl>\nADDLIB $<TARGET_FILE:pccl>\n${_mri_addlibs}SAVE\nEND\n")
        add_custom_command(TARGET pccl POST_BUILD
            COMMAND ${CMAKE_AR} -M < "${_mri_file}"
            COMMENT "Merging component archives into fat libpccl.a"
        )
    else ()
        # macOS: libtool -static
        add_custom_command(TARGET pccl POST_BUILD
            COMMAND libtool -static -o "$<TARGET_FILE:pccl>"
                "$<TARGET_FILE:pccl>" ${_pccl_fat_libs}
            COMMENT "Merging component archives into fat libpccl.a (macOS)"
        )
    endif ()
endif ()

if (PCCL_BUILD_TESTS)
    add_subdirectory(tests)
endif ()

# ---------------------------------------------------------------------------
# Install rules, cmake package config, and pkg-config
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS pccl
    EXPORT pcclTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ccoip/public_include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Generate and install cmake package config
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pcclConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/pcclConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pccl"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pcclConfig.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pccl"
)
install(EXPORT pcclTargets
    FILE pcclTargets.cmake
    NAMESPACE pccl::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pccl"
)

# Generate and install pkg-config file (for non-cmake consumers: cc main.c -lpccl $(pkg-config --libs pccl))
set(_pccl_pc_libs "")
if (PCCL_BUILD_HIP_SUPPORT AND hip_FOUND)
    set(_pccl_pc_libs "${_pccl_pc_libs} -lamdhip64")
    get_target_property(_hip_lib_dir hip::host LOCATION)
    if (_hip_lib_dir)
        get_filename_component(_hip_lib_dir "${_hip_lib_dir}" DIRECTORY)
        set(_pccl_pc_libs "-L${_hip_lib_dir} ${_pccl_pc_libs}")
    endif ()
endif ()
if (PCCL_BUILD_CUDA_SUPPORT AND CUDAToolkit_FOUND)
    set(_pccl_pc_libs "${_pccl_pc_libs} -lcuda")
endif ()
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pccl.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/pccl.pc"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pccl.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)
