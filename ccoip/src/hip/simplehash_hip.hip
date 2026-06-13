// HIP port of simplehash_cuda.cu
// TODO: OPTIMIZE THIS FURTHER (AND KEEP IN SYNC WITH CPU IMPLEMENTATION)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

/**
 * @brief Functor to combine two hash values.
 *
 * This is a simple hash combine function that uses a prime number multiplication.
 */
struct HashCombine {
    __host__ __device__ uint32_t operator()(uint32_t a, uint32_t b) const {
        return a ^ (b + 0x9e3779b9u + (a << 6) + (a >> 2));
    }
};

/**
 * @brief Warp-level reduction using shuffle-down intrinsics.
 *
 * Reduces all lanes in a warp to lane 0 using XOR-based hash combination.
 * HIP does not use a mask parameter — wavefront width is always 32 for RDNA targets
 * compiled with -mwavefrontsize32.
 */
__device__ uint32_t warpReduceHash(uint32_t val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = HashCombine()(val, __shfl_down(val, offset));
    }
    return val;
}

/**
 * @brief Block-level reduction.
 *
 * Each warp reduces to its lane-0, writes to shared memory, then the first warp
 * reduces across all warp results.
 */
__device__ uint32_t blockReduceHash(uint32_t val) {
    __shared__ uint32_t warpResults[32];

    const int lane = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;

    val = warpReduceHash(val);

    if (lane == 0) {
        warpResults[warpId] = val;
    }
    __syncthreads();

    const int numWarps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < numWarps) ? warpResults[lane] : 0u;

    if (warpId == 0) {
        val = warpReduceHash(val);
    }
    return val;
}

/**
 * @brief First-pass kernel: vectorised uint4 loads, one partial result per block.
 *
 * Each block processes an equal-sized stripe of the input, producing a single
 * 32-bit partial hash that is stored to d_partial[blockIdx.x].
 */
__global__ void hip_big_reduce_kernel(const uint32_t *__restrict d_in,
                                      uint32_t *__restrict d_partial,
                                      const size_t n_words) {
    const auto *d4 = reinterpret_cast<const uint4 *>(d_in);
    const auto totalVec = static_cast<unsigned int>(n_words / 4);

    const uint32_t elementsPerBlock = (totalVec + gridDim.x - 1) / gridDim.x;
    const uint32_t blockStart = blockIdx.x * elementsPerBlock;
    const uint32_t blockEnd = min(blockStart + elementsPerBlock, totalVec);

    uint32_t threadVal = 0;
    for (uint32_t i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x) {
        const uint4 v = d4[i];
        threadVal = HashCombine()(threadVal, v.x);
        threadVal = HashCombine()(threadVal, v.y);
        threadVal = HashCombine()(threadVal, v.z);
        threadVal = HashCombine()(threadVal, v.w);
    }

    threadVal = blockReduceHash(threadVal);
    if (threadIdx.x == 0) {
        d_partial[blockIdx.x] = threadVal;
    }
}

/**
 * @brief Second-pass kernel: reduces all partial results to a single value.
 *
 * Launched with a single block of blockDim threads over gridDim_x partial results.
 */
__global__ void hip_final_reduce_kernel(const uint32_t *__restrict d_partial,
                                        uint32_t *__restrict d_out,
                                        const uint32_t gridDim_x) {
    uint32_t val = 0;
    for (uint32_t i = threadIdx.x; i < gridDim_x; i += blockDim.x) {
        val = HashCombine()(val, d_partial[i]);
    }
    val = blockReduceHash(val);
    if (threadIdx.x == 0) {
        *d_out = val;
    }
}

/**
 * @brief Host entry point: compute a 32-bit rolling hash over a device buffer.
 *
 * Returns 0 on any HIP error (prints a diagnostic to stderr) rather than
 * calling abort(), so callers can detect failure without crashing.
 *
 * @param data     Device pointer to the input data (must be 16-byte aligned).
 * @param n_bytes  Number of bytes to hash.
 * @return 32-bit hash value, zero-extended to 64 bits.  Returns 0 on error.
 */
extern "C" uint64_t simplehash_hip_kernel(const void *data, const size_t n_bytes) {
    if (n_bytes == 0) {
        return 0;
    }
    if (reinterpret_cast<uint64_t>(data) % 16 != 0) {
        fprintf(stderr, "[simplehash_hip] input pointer is not 16-byte aligned\n");
        return 0;
    }

    // Flush any stale async error from a previous operation so that our own
    // hipGetLastError() calls below only reflect errors from THIS invocation.
    (void)hipGetLastError();

    const size_t n_words = n_bytes / sizeof(uint32_t);

    void *d_out = nullptr;
    if (hipMalloc(&d_out, sizeof(uint32_t)) != hipSuccess) {
        fprintf(stderr, "[simplehash_hip] hipMalloc(d_out) failed\n");
        return 0;
    }

    uint32_t final_host_val = 0;

    if (n_words > 0) {
        const size_t n_vec = n_words / 4;
        const auto tail_ints = static_cast<uint32_t>(n_words % 4);

        if (n_vec > 0) {
            constexpr int blockDim_x = 256;
            auto gridDim_x = static_cast<uint32_t>((n_vec + blockDim_x - 1) / blockDim_x);
            if (gridDim_x > 960) {
                gridDim_x = 960;
            }

            void *d_partial = nullptr;
            if (hipMalloc(&d_partial, gridDim_x * sizeof(uint32_t)) != hipSuccess) {
                fprintf(stderr, "[simplehash_hip] hipMalloc(d_partial) failed\n");
                hipFree(d_out);
                return 0;
            }

            const auto *d_in_ptr = static_cast<const uint32_t *>(data);
            auto *d_partial_ptr = static_cast<uint32_t *>(d_partial);

            hip_big_reduce_kernel<<<gridDim_x, blockDim_x>>>(d_in_ptr, d_partial_ptr, n_words);
            // Check for kernel launch errors before issuing the second kernel.
            hipError_t err = hipGetLastError();
            if (err != hipSuccess) {
                fprintf(stderr, "[simplehash_hip] big_reduce_kernel launch failed: %s\n",
                        hipGetErrorString(err));
                hipFree(d_partial);
                hipFree(d_out);
                return 0;
            }

            hip_final_reduce_kernel<<<1, blockDim_x>>>(d_partial_ptr,
                                                       static_cast<uint32_t *>(d_out),
                                                       gridDim_x);
            err = hipGetLastError();
            if (err != hipSuccess) {
                fprintf(stderr, "[simplehash_hip] final_reduce_kernel launch failed: %s\n",
                        hipGetErrorString(err));
                hipFree(d_partial);
                hipFree(d_out);
                return 0;
            }

            // Wait for both kernels to finish; this also surfaces any GPU-side faults.
            err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                fprintf(stderr, "[simplehash_hip] hipDeviceSynchronize failed: %s\n",
                        hipGetErrorString(err));
                hipFree(d_partial);
                hipFree(d_out);
                return 0;
            }

            if (hipMemcpy(&final_host_val, d_out, sizeof(uint32_t), hipMemcpyDeviceToHost) != hipSuccess) {
                fprintf(stderr, "[simplehash_hip] hipMemcpy(result DtoH) failed\n");
                hipFree(d_partial);
                hipFree(d_out);
                return 0;
            }

            hipFree(d_partial);
        }

        if (tail_ints > 0) {
            uint32_t tail_host[4] = {0, 0, 0, 0};
            const void *tail_ptr = static_cast<const char *>(data) + n_vec * 4 * sizeof(uint32_t);
            if (hipMemcpy(tail_host, tail_ptr, tail_ints * sizeof(uint32_t),
                          hipMemcpyDeviceToHost) != hipSuccess) {
                fprintf(stderr, "[simplehash_hip] hipMemcpy(tail_ints DtoH) failed\n");
                hipFree(d_out);
                return 0;
            }
            for (size_t i = 0; i < tail_ints; ++i) {
                final_host_val = HashCombine()(final_host_val, tail_host[i]);
            }
        }
    }

    const size_t tail_bytes = n_bytes % sizeof(uint32_t);
    if (tail_bytes > 0) {
        const void *tail_ptr = static_cast<const char *>(data) + n_words * sizeof(uint32_t);
        uint8_t tail_host[4] = {0, 0, 0, 0};
        if (hipMemcpy(tail_host, tail_ptr, tail_bytes, hipMemcpyDeviceToHost) != hipSuccess) {
            fprintf(stderr, "[simplehash_hip] hipMemcpy(tail_bytes DtoH) failed\n");
            hipFree(d_out);
            return 0;
        }
        uint32_t tailVal = 0;
        for (size_t i = 0; i < tail_bytes; ++i) {
            tailVal |= static_cast<uint32_t>(tail_host[i]) << (8 * i);
        }
        final_host_val = HashCombine()(final_host_val, tailVal);
    }

    hipFree(d_out);
    return final_host_val;
}
