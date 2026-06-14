// HIP port of simplehash_cuda.cu
// TODO: OPTIMIZE THIS FURTHER (AND KEEP IN SYNC WITH CPU IMPLEMENTATION)

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// amdgpu_no_* attributes: tell the compiler NOT to add the corresponding
// hidden implicit argument to the kernel descriptor.  The most critical one
// is amdgpu_no_hostcall_ptr: without it the HIP runtime calls
// amd_comgr_do_action to set up the hostcall service which crashes on
// gfx1201 in ROCm 7.2.4.  All others are similarly unused in our kernels.
// Guard with __AMDGCN__ so the host-compilation pass never sees them.
// ---------------------------------------------------------------------------
#if defined(__AMDGCN__)
#  define PCCL_AMDGPU_KERNEL_ATTRS \
    __attribute__(( \
        amdgpu_no_hostcall_ptr,      \
        amdgpu_no_heap_ptr,          \
        amdgpu_no_multigrid_sync_arg, \
        amdgpu_no_default_queue,     \
        amdgpu_no_dispatch_ptr,      \
        amdgpu_no_queue_ptr,         \
        amdgpu_no_dispatch_id,       \
        amdgpu_no_lds_kernel_id      \
    ))
#else
#  define PCCL_AMDGPU_KERNEL_ATTRS
#endif

/**
 * @brief Functor to combine two hash values.
 *
 * Must match the CUDA implementation in simplehash_cuda.cu exactly.
 */
struct HashCombine {
    __host__ __device__ __forceinline__ uint32_t operator()(uint32_t a, const uint32_t b) const {
        a ^= b + 0x9e3779b1u;
        a = (a << 7) | (a >> 25);
        a *= 0x85ebca6bu;
        return a;
    }
};

/**
 * @brief Warp-level reduction using shuffle-down intrinsics (wavefront size 32).
 */
__device__ __forceinline__ uint32_t warpReduceHash(uint32_t val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = HashCombine()(val, __shfl_down(val, offset));
    }
    return val;
}

/**
 * @brief Block-level reduction (blockDim must be a multiple of 32, max 256).
 */
__device__ __forceinline__ uint32_t blockReduceHash(uint32_t val) {
    __shared__ uint32_t warpResults[8]; // one slot per warp (256/32 = 8 max)

    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;

    val = warpReduceHash(val);

    if (lane == 0) {
        warpResults[warpId] = val;
    }
    __syncthreads();

    const int numWarps = blockDim.x >> 5;  // blockDim.x / 32
    val = (threadIdx.x < numWarps) ? warpResults[threadIdx.x] : 0u;

    if (warpId == 0) {
        val = warpReduceHash(val);
    }
    return val;
}

/**
 * @brief First-pass kernel: one partial result per block, uint4 vectorized loads.
 *
 * Matches the CUDA big_reduce_kernel exactly: interprets the input as uint4 elements
 * (4 x uint32), divides them evenly among blocks, and each thread strides through
 * its block's share combining all four components per vector element.
 */
__global__ PCCL_AMDGPU_KERNEL_ATTRS void hip_big_reduce_kernel(const uint32_t * d_in,
                                      uint32_t       * d_partial,
                                      const uint64_t   n_words) {
    const auto * d4 = reinterpret_cast<const uint4 *>(d_in);
    const auto totalVec = static_cast<uint32_t>(n_words / 4);

    const uint32_t elementsPerBlock = (totalVec + gridDim.x - 1) / gridDim.x;
    const uint32_t blockStart = blockIdx.x * elementsPerBlock;
    uint32_t blockEnd = blockStart + elementsPerBlock;
    if (blockEnd > totalVec) {
        blockEnd = totalVec;
    }

    uint32_t sum = 0u;
    for (uint32_t i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x) {
        const uint4 v = d4[i];
        sum = HashCombine()(sum, v.x);
        sum = HashCombine()(sum, v.y);
        sum = HashCombine()(sum, v.z);
        sum = HashCombine()(sum, v.w);
    }

    sum = blockReduceHash(sum);
    if (threadIdx.x == 0) {
        d_partial[blockIdx.x] = sum;
    }
}

/**
 * @brief Second-pass kernel: reduces partial results to a single value.
 */
__global__ PCCL_AMDGPU_KERNEL_ATTRS void hip_final_reduce_kernel(const uint32_t * d_partial,
                                        uint32_t       * d_out,
                                        const uint32_t   numPartials) {
    uint32_t val = 0u;
    for (uint32_t i = threadIdx.x; i < numPartials; i += blockDim.x) {
        val = HashCombine()(val, d_partial[i]);
    }
    val = blockReduceHash(val);
    if (threadIdx.x == 0) {
        *d_out = val;
    }
}

/**
 * @brief Host entry point: 32-bit rolling hash over a device buffer.
 *
 * Matches the CUDA implementation in simplehash_cuda.cu exactly:
 * - Vectorised uint4 first pass (n_vec = n_words/4 elements).
 * - Tail uint32 words (n_words % 4) hashed on the host.
 * - Tail bytes (n_bytes % 4) hashed on the host.
 *
 * Returns 0 on any HIP error (prints to stderr) rather than calling abort().
 *
 * @param data     Device pointer (from hipMalloc), must be 16-byte aligned.
 * @param n_bytes  Number of bytes to hash.
 * @return 32-bit hash, zero-extended to 64 bits. Returns 0 on error.
 */
extern "C" uint64_t simplehash_hip_kernel(const void * data,
                                          const size_t n_bytes) {
    if (n_bytes == 0) {
        return 0;
    }
    if (data == nullptr) {
        fprintf(stderr, "[simplehash_hip] data pointer is null\n");
        return 0;
    }
    if (reinterpret_cast<uintptr_t>(data) % 16 != 0) {
        fprintf(stderr, "[simplehash_hip] input pointer is not 16-byte aligned\n");
        return 0;
    }

    // Flush any stale async error before we issue new work.
    (void)hipGetLastError();

    const size_t n_words = n_bytes / sizeof(uint32_t);
    const size_t n_vec   = n_words / 4;        // number of uint4 elements
    const auto tail_ints = static_cast<uint32_t>(n_words % 4); // leftover words
    const size_t tail_bytes = n_bytes % sizeof(uint32_t);

    uint32_t final_host_val = 0u;

    if (n_vec > 0) {
        void * d_out     = nullptr;
        void * d_partial = nullptr;

        if (hipMalloc(&d_out, sizeof(uint32_t)) != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] hipMalloc(d_out) failed\n");
            return 0;
        }
        if (hipMemset(d_out, 0, sizeof(uint32_t)) != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] hipMemset(d_out) failed\n");
            hipFree(d_out);
            return 0;
        }

        constexpr uint32_t blockDim_x = 256;
        uint32_t gridDim_x = static_cast<uint32_t>((n_vec + blockDim_x - 1) / blockDim_x);
        if (gridDim_x > 960) {
            gridDim_x = 960;
        }

        if (hipMalloc(&d_partial, gridDim_x * sizeof(uint32_t)) != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] hipMalloc(d_partial) failed\n");
            hipFree(d_out);
            return 0;
        }

        const auto * d_in_ptr      = static_cast<const uint32_t *>(data);
        auto       * d_partial_ptr = static_cast<uint32_t *>(d_partial);
        auto       * d_out_ptr     = static_cast<uint32_t *>(d_out);

        hip_big_reduce_kernel<<<gridDim_x, blockDim_x>>>(
            d_in_ptr, d_partial_ptr, static_cast<uint64_t>(n_words));
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] big_reduce_kernel launch failed: %s\n",
                    hipGetErrorString(err));
            hipFree(d_partial);
            hipFree(d_out);
            return 0;
        }

        hip_final_reduce_kernel<<<1, blockDim_x>>>(
            d_partial_ptr, d_out_ptr, gridDim_x);
        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] final_reduce_kernel launch failed: %s\n",
                    hipGetErrorString(err));
            hipFree(d_partial);
            hipFree(d_out);
            return 0;
        }

        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] hipDeviceSynchronize failed: %s\n",
                    hipGetErrorString(err));
            hipFree(d_partial);
            hipFree(d_out);
            return 0;
        }

        if (hipMemcpy(&final_host_val, d_out, sizeof(uint32_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] hipMemcpy(result DtoH) failed\n");
            hipFree(d_partial);
            hipFree(d_out);
            return 0;
        }

        hipFree(d_partial);
        hipFree(d_out);
    }

    // Hash leftover uint32 words that didn't form a complete uint4.
    if (tail_ints > 0) {
        uint32_t tail_host[4] = {0u, 0u, 0u, 0u};
        const void * tail_ptr =
            static_cast<const char *>(data) + n_vec * 4 * sizeof(uint32_t);
        if (hipMemcpy(tail_host, tail_ptr, tail_ints * sizeof(uint32_t),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] hipMemcpy(tail_ints DtoH) failed\n");
            return 0;
        }
        for (uint32_t i = 0; i < tail_ints; ++i) {
            final_host_val = HashCombine()(final_host_val, tail_host[i]);
        }
    }

    // Hash any remaining sub-word bytes.
    if (tail_bytes > 0) {
        const void * tail_ptr =
            static_cast<const char *>(data) + n_words * sizeof(uint32_t);
        uint8_t tail_host[4] = {0, 0, 0, 0};
        if (hipMemcpy(tail_host, tail_ptr, tail_bytes,
                      hipMemcpyDeviceToHost) != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] hipMemcpy(tail_bytes DtoH) failed\n");
            return 0;
        }
        uint32_t tailVal = 0u;
        for (size_t i = 0; i < tail_bytes; ++i) {
            tailVal |= static_cast<uint32_t>(tail_host[i]) << (8u * i);
        }
        final_host_val = HashCombine()(final_host_val, tailVal);
    }

    return final_host_val;
}
