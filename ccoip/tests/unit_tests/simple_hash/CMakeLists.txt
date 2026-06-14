# test simplehash cuda hash
if (PCCL_BUILD_CUDA_SUPPORT)
    # NOTE: CUDA doesn't work with ASAN, hence use un-sanitized tests here
    add_unsanitized_gtest(simplehash_cuda_test simplehash_cuda_test.cpp)
    target_link_libraries(simplehash_cuda_test PRIVATE simplehash_cuda)
    target_link_libraries(simplehash_cuda_test PRIVATE ccoip_private CUDA::cuda_driver)

    # test simplehash cpu hash
    add_unsanitized_gtest(simplehash_cpu_test simplehash_cpu_test.cu)
    target_link_libraries(simplehash_cpu_test PRIVATE simplehash_cpu)
    target_link_libraries(simplehash_cpu_test PRIVATE simplehash_cuda)
    target_link_libraries(simplehash_cpu_test PRIVATE ccoip_private CUDA::cudart)
endif ()


# test simplehash cpu hash
add_sanitized_gtest(simplehash_cpu_no_cuda_test simplehash_cpu_test_no_cuda.cpp)
target_link_libraries(simplehash_cpu_no_cuda_test PRIVATE ccoip_private simplehash_cpu)

if (PCCL_BUILD_HIP_SUPPORT)
    add_unsanitized_gtest(simplehash_hip_test simplehash_hip_test.cpp)
    target_link_libraries(simplehash_hip_test PRIVATE simplehash_hip)
    target_link_libraries(simplehash_hip_test PRIVATE ccoip_private hip::host)

    # Remove the conflicting system libamdhip64.so.5 NEEDED entry from the binary.
    # When --hip-link is used, the system libamdhip64.so (SONAME .so.5) is picked up
    # via /lib/x86_64-linux-gnu before /opt/rocm-*/lib, causing both .so.5 and .so.7
    # to be loaded simultaneously, which corrupts the HIP runtime state.
    find_program(PATCHELF_EXECUTABLE patchelf)
    if (PATCHELF_EXECUTABLE)
        add_custom_command(TARGET simplehash_hip_test POST_BUILD
            COMMAND ${PATCHELF_EXECUTABLE} --remove-needed libamdhip64.so.5
                    $<TARGET_FILE:simplehash_hip_test>
            COMMENT "Removing conflicting libamdhip64.so.5 NEEDED entry from simplehash_hip_test"
        )
    endif ()
endif ()

