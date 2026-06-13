#include <chrono>
#include <cstring>
#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

extern "C" uint64_t simplehash_hip_kernel(const void *data, size_t n_bytes);

#define HIP_CHECK(call) \
    do { \
        hipError_t _e = (call); \
        ASSERT_EQ(hipSuccess, _e) << "HIP error: " << hipGetErrorString(_e); \
    } while (0)

uint32_t next_uint32(uint32_t &seed, const uint32_t lo, const uint32_t hi) {
    seed = 1664525u * seed + 1013904223u;
    const uint64_t range = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo) + 1ULL;
    return static_cast<uint32_t>(static_cast<uint64_t>(seed) % range) + lo;
}

TEST(SimpleHashHipTest, BenchmarkAgainstBaseline) {
    std::vector<uint8_t> data(154533888, 0);

    // init random
    {
        uint32_t seed = 42;
        for (uint32_t i = 0; i < data.size(); i++) {
            data[i] = next_uint32(seed, 0, 255);
        }
    }

    constexpr int n_repeat = 100;

    volatile uint64_t simple_hashes[n_repeat]{};
    void *data_ptr = nullptr;
    HIP_CHECK(hipMalloc(&data_ptr, data.size()));
    HIP_CHECK(hipMemcpy(data_ptr, data.data(), data.size(), hipMemcpyHostToDevice));

    // launch benchmark
    {
        const auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_repeat; ++i) {
            simple_hashes[i] = simplehash_hip_kernel(data_ptr, data.size());
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "HIP: " << duration.count() << " us" << std::endl;
        const double bandwidth = (static_cast<double>(n_repeat * data.size()) / 1e9) /
                                 (static_cast<double>(duration.count()) / 1e6);
        std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

        // assert all hashes are the same
        for (int i = 1; i < n_repeat; ++i) {
            ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
        }
    }
    hipFree(data_ptr);

    for (int i = 0; i < n_repeat; ++i) {
        ASSERT_EQ(3391090508u, simple_hashes[i]);
    }
}

TEST(SimpleHashHipTest, TestSizeOneByte) {
    std::vector<uint8_t> data(1, 0);
    {
        uint32_t seed = 42;
        for (uint32_t i = 0; i < data.size(); i++) {
            data[i] = next_uint32(seed, 0, 255);
        }
    }

    void *data_ptr = nullptr;
    HIP_CHECK(hipMalloc(&data_ptr, data.size()));
    HIP_CHECK(hipMemcpy(data_ptr, data.data(), data.size(), hipMemcpyHostToDevice));

    const uint64_t hash = simplehash_hip_kernel(data_ptr, data.size());
    hipFree(data_ptr);

    ASSERT_NE(0u, hash);
}

TEST(SimpleHashHipTest, TestSizeZero) {
    const uint64_t hash = simplehash_hip_kernel(nullptr, 0);
    ASSERT_EQ(0u, hash);
}

TEST(SimpleHashHipTest, TestDeterminism) {
    std::vector<uint8_t> data(4096, 0);
    {
        uint32_t seed = 123;
        for (uint32_t i = 0; i < data.size(); i++) {
            data[i] = next_uint32(seed, 0, 255);
        }
    }

    void *data_ptr = nullptr;
    HIP_CHECK(hipMalloc(&data_ptr, data.size()));
    HIP_CHECK(hipMemcpy(data_ptr, data.data(), data.size(), hipMemcpyHostToDevice));

    const uint64_t hash1 = simplehash_hip_kernel(data_ptr, data.size());
    const uint64_t hash2 = simplehash_hip_kernel(data_ptr, data.size());
    hipFree(data_ptr);

    ASSERT_EQ(hash1, hash2);
    ASSERT_NE(0u, hash1);
}

int main(int argc, char **argv) {
    hipError_t initErr = hipSetDevice(0);
    if (initErr != hipSuccess) {
        std::cerr << "hipSetDevice(0) failed: " << hipGetErrorString(initErr) << std::endl;
        return 1;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
