#include <hash_utils.hpp>
#include <pccl_log.hpp>
#include <cstddef>

#ifdef PCCL_HAS_HIP_SUPPORT
extern "C" uint64_t simplehash_hip_kernel(const void *data, size_t n_bytes);
#endif

uint32_t ccoip::hash_utils::simplehash_hip(const void *data, const size_t n_bytes) {
#ifndef PCCL_HAS_HIP_SUPPORT
    LOG(BUG) << "simplehash_hip called but PCCL is not built with HIP support. This is a bug.";
    return 0;
#else
    return static_cast<uint32_t>(simplehash_hip_kernel(data, n_bytes));
#endif
}
