// HIP port of simplehash_cuda.cu
// TODO: OPTIMIZE THIS FURTHER (AND KEEP IN SYNC WITH CPU IMPLEMENTATION)

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

/**
 * @brief Functor to combine two hash values.
 */
struct HashCombine {
    __host__ __device__ uint32_t operator()(uint32_t a, uint32_t b) const {
        return a ^ (b + 0x9e3779b9u + (a << 6) + (a >> 2));
    }
};

/**
 * @brief Warp-level reduction using shuffle-down intrinsics (wavefront size 32).
 */
__device__ uint32_t warpReduceHash(uint32_t val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = HashCombine()(val, __shfl_down(val, offset));
    }
    return val;
}

/**
 * @brief Block-level reduction (blockDim must be a multiple of 32, max 256).
 */
__device__ uint32_t blockReduceHash(uint32_t val) {
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
 * @brief First-pass kernel: one partial result per block, scalar uint32 loads.
 *
 * Using uint64_t for n_words avoids HIP issues with size_t kernel parameters.
 * Using scalar loads avoids any potential uint4 alignment/vectorisation faults.
 */
__global__ void hip_big_reduce_kernel(const uint32_t * d_in,
                                      uint32_t       * d_partial,
                                      const uint64_t   n_words) {
    const uint64_t wordsPerBlock =
        (n_words + (uint64_t)gridDim.x - 1) / (uint64_t)gridDim.x;
    const uint64_t blockStart = (uint64_t)blockIdx.x * wordsPerBlock;
    const uint64_t blockEndUnclamped = blockStart + wordsPerBlock;
    const uint64_t blockEnd = (blockEndUnclamped < n_words) ? blockEndUnclamped : n_words;

    uint32_t threadVal = 0u;
    for (uint64_t i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x) {
        threadVal = HashCombine()(threadVal, d_in[i]);
    }

    threadVal = blockReduceHash(threadVal);
    if (threadIdx.x == 0) {
        d_partial[blockIdx.x] = threadVal;
    }
}

/**
 * @brief Second-pass kernel: reduces partial results to a single value.
 */
__global__ void hip_final_reduce_kernel(const uint32_t * d_partial,
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
 * Returns 0 on any HIP error (prints to stderr) rather than calling abort().
 *
 * @param data     Device pointer (from hipMalloc), must be 4-byte aligned.
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
    if (reinterpret_cast<uintptr_t>(data) % 4 != 0) {
        fprintf(stderr, "[simplehash_hip] input pointer is not 4-byte aligned\n");
        return 0;
    }

    // Flush any stale async error before we issue new work.
    (void)hipGetLastError();

    const uint64_t n_words = n_bytes / sizeof(uint32_t);

    void    * d_out     = nullptr;
    void    * d_partial = nullptr;
    uint32_t  final_host_val = 0u;

    if (hipMalloc(&d_out, sizeof(uint32_t)) != hipSuccess) {
        fprintf(stderr, "[simplehash_hip] hipMalloc(d_out) failed\n");
        return 0;
    }
    if (hipMemset(d_out, 0, sizeof(uint32_t)) != hipSuccess) {
        fprintf(stderr, "[simplehash_hip] hipMemset(d_out) failed\n");
        hipFree(d_out);
        return 0;
    }

    if (n_words > 0) {
        constexpr uint32_t blockDim_x = 256;
        uint32_t gridDim_x = static_cast<uint32_t>((n_words + blockDim_x - 1) / blockDim_x);
        if (gridDim_x > 960) {
            gridDim_x = 960;
        }

        if (hipMalloc(&d_partial, gridDim_x * sizeof(uint32_t)) != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] hipMalloc(d_partial) failed\n");
            hipFree(d_out);
            return 0;
        }
        if (hipMemset(d_partial, 0, gridDim_x * sizeof(uint32_t)) != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] hipMemset(d_partial) failed\n");
            hipFree(d_partial);
            hipFree(d_out);
            return 0;
        }

        const auto * d_in_ptr      = static_cast<const uint32_t *>(data);
        auto       * d_partial_ptr = static_cast<uint32_t *>(d_partial);
        auto       * d_out_ptr     = static_cast<uint32_t *>(d_out);

        hip_big_reduce_kernel<<<gridDim_x, blockDim_x>>>(
            d_in_ptr, d_partial_ptr, n_words);
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
    }

    // Handle trailing uint32 words (n_bytes not a multiple of 16)
    const uint64_t tail_ints = n_bytes / sizeof(uint32_t) - n_words; // always 0 since n_words = n_bytes/4
    // Note: tail_ints is always 0 because n_words = n_bytes / sizeof(uint32_t)
    // The remaining sub-word bytes are handled below.

    const size_t tail_bytes = n_bytes % sizeof(uint32_t);
    if (tail_bytes > 0) {
        const void * tail_ptr =
            static_cast<const char *>(data) + n_words * sizeof(uint32_t);
        uint8_t tail_host[4] = {0, 0, 0, 0};
        if (hipMemcpy(tail_host, tail_ptr, tail_bytes,
                      hipMemcpyDeviceToHost) != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] hipMemcpy(tail_bytes DtoH) failed\n");
            hipFree(d_out);
            return 0;
        }
        uint32_t tailVal = 0u;
        for (size_t i = 0; i < tail_bytes; ++i) {
            tailVal |= static_cast<uint32_t>(tail_host[i]) << (8u * i);
        }
        final_host_val = HashCombine()(final_host_val, tailVal);
    }

    hipFree(d_out);
    return final_host_val;
}
