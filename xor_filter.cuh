#ifndef FOURWISE_XOR_BINARY_FUSE_FILTER_XOR_FILTER_LOWMEM_H_
#define FOURWISE_XOR_BINARY_FUSE_FILTER_XOR_FILTER_LOWMEM_H_

#include <vector>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <string>
#ifdef _WIN32
#include "uint128_t.cuh"
#else // _WIN32
#include "uint128_t_linux.cuh"
#endif

#include <random>
#include <cuda_runtime.h>

using namespace std;
#define XOR_MAX_SIZE SIZE_MAX



class SimpleMixSplit {

public:
    __device__ __host__ SimpleMixSplit() {
    }

    __device__ __host__ inline static uint64_t murmur64(uint64_t h) {
        h ^= h >> 33;
        h *= UINT64_C(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h *= UINT64_C(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        return h;
    }
    __device__ __host__ inline static uint64_t murmur128(uint128_t h) {
        h ^= h >> 33;
        h *= UINT64_C(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h *= UINT64_C(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        return h;
    }
    __device__ __host__ inline uint64_t operator()(uint64_t key) const {
        return murmur64(key);
    }
    __device__ __host__ inline uint128_t operator()(uint128_t key) const {

        return murmur128(key);
    }
};



__device__ __host__ static inline size_t calculateSegmentLength(size_t arity, size_t size) {
    size_t segmentLength;
    double baseLog = std::log(3.33);
    if (arity == 3) {
#ifdef __CUDA_ARCH__
        segmentLength = size_t(1) << (int)floor(log((double)size) / baseLog + 2.25);
#else
        segmentLength = size_t(1) << (int)floor(std::log((double)size) / baseLog + 2.25);
#endif
    }
    else if (arity == 4) {
        baseLog = std::log(2.91);
#ifdef __CUDA_ARCH__
        segmentLength = size_t(1) << (int)floor(log((double)size) / baseLog - 0.5);
#else
        segmentLength = size_t(1) << (int)floor(std::log((double)size) / baseLog - 0.5);
#endif
    }
    else {
        segmentLength = 65536;
    }
    return segmentLength;
}

__device__ __host__ static inline double calculateSizeFactor(size_t arity, size_t size) {
    if (size <= 2) { size = 2; }
    double sizeFactor;
    double baseLogSize = size > 0 ? log((double)size) : 0;
    const double log1000000 = log(1000000.0);
    const double log600000 = log(600000.0);

    if (arity == 3) {
#ifdef __CUDA_ARCH__
        sizeFactor = fmax(1.125, 0.875 + 0.25 * log1000000 / baseLogSize);
#else
        sizeFactor = std::max(1.125, 0.875 + 0.25 * log1000000 / baseLogSize);
#endif
    }
    else if (arity == 4) {
#ifdef __CUDA_ARCH__
        sizeFactor = fmax(1.075, 0.77 + 0.305 * log600000 / baseLogSize);
#else
        sizeFactor = std::max(1.075, 0.77 + 0.305 * log600000 / baseLogSize);
#endif
    }
    else {
        sizeFactor = 2.0;
    }
    return sizeFactor;
}

/**
 * As of July 2021, the lowmem versions of the binary fuse filters are
 * the recommended defaults.
 */
namespace xorbinaryfusefilter_lowmem4wise {

    enum Status {
        Ok = 0,
        NotFound = 1,
        NotEnoughSpace = 2,
        NotSupported = 3,
    };

    __device__ __host__ inline uint32_t reduce(uint32_t hash, uint32_t n) {
        return (uint32_t)(((uint64_t)hash * n) >> 32);
    }

    template <typename ItemType, typename FingerprintType, typename HashFamily = SimpleMixSplit>
    class XorBinaryFuseFilter {
    public:
        size_t size;
        size_t arrayLength;
        size_t segmentCount;
        size_t segmentCountLength;
        size_t segmentLength;
        size_t segmentLengthMask;
        static constexpr size_t arity = 4;
        FingerprintType* fingerprints;
        HashFamily* hasher;
        size_t hashIndex{ 0 };
        uint64_t Seed;
        __device__ __host__ static inline FingerprintType fingerprint(uint64_t hash)
        {
            return hash ^ (hash >> 32);
        }

        __device__ __host__ static inline uint64_t rng_splitmix64(uint64_t* seed)
        {
            uint64_t z = (*seed += UINT64_C(0x9E3779B97F4A7C15));
            z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
            return z ^ (z >> 31);
        }


        __device__ __host__ static inline uint64_t rotateLeft(uint64_t n, unsigned int c) {
            const unsigned int mask = (CHAR_BIT * sizeof(n) - 1);
            c &= mask;
            return (n << c) | (n >> ((-c) & mask));
        }

        __device__ __host__ static inline uint64_t rotateRight(uint64_t n, unsigned int c) {
            const unsigned int mask = (CHAR_BIT * sizeof(n) - 1);
            c &= mask;
            return (n >> c) | (n << ((-c) & mask));
        }

        __device__ __host__ inline uint64_t getHashFromHash(uint64_t hash, int index)  const
        {
            uint128_t x = (uint128_t)hash * (uint128_t)segmentCountLength;
            uint64_t h = (uint64_t)(x >> 64);
            h += index * segmentLength;
            uint64_t hh = hash & ((UINT64_C(1) << 36) - 1);
            h ^= (size_t)((hh >> (36 - 18 * index)) & segmentLengthMask);
            return h;
        }



        __device__ __host__ explicit XorBinaryFuseFilter() {
            uint64_t rng_counter = 0x726b2b9d438b9d4d;
            Seed = rng_splitmix64(&rng_counter);
            size_t size = 1;
            hasher = new HashFamily();
            this->size = size;
            this->segmentLength = calculateSegmentLength(arity, size);
            if (this->segmentLength > (1 << 18)) {
                this->segmentLength = (1 << 18);
            }
            double sizeFactor = calculateSizeFactor(arity, size);
            size_t capacity = static_cast<size_t>(size * sizeFactor);
            size_t segmentCount = (capacity + segmentLength - 1) / segmentLength - (arity - 1);
            this->arrayLength = (segmentCount + arity - 1) * segmentLength;
            this->segmentLengthMask = this->segmentLength - 1;
            this->segmentCount = (this->arrayLength + this->segmentLength - 1) / this->segmentLength;
            this->segmentCount = this->segmentCount <= arity - 1 ? 1 : this->segmentCount - (arity - 1);
            this->arrayLength = (this->segmentCount + arity - 1) * this->segmentLength;
            this->segmentCountLength = this->segmentCount * this->segmentLength;
            fingerprints = new FingerprintType[arrayLength]();
            memset(fingerprints, 0, sizeof(FingerprintType) * arrayLength);
        }


        __device__ __host__ explicit XorBinaryFuseFilter(const size_t sizes) {
            uint64_t rng_counter = 0x726b2b9d438b9d4d;
            Seed = rng_splitmix64(&rng_counter);
            size_t size = sizes;
            if (size < 99999)
            {
                size = 99999;
            }
            hasher = new HashFamily();
            this->size = size;
            this->segmentLength = calculateSegmentLength(arity, size);
            if (this->segmentLength > (1 << 18)) {
                this->segmentLength = (1 << 18);
            }
            double sizeFactor = calculateSizeFactor(arity, size);
            size_t capacity = static_cast<size_t>(size * sizeFactor);
            size_t segmentCount = (capacity + segmentLength - 1) / segmentLength - (arity - 1);
            this->arrayLength = (segmentCount + arity - 1) * segmentLength;
            this->segmentLengthMask = this->segmentLength - 1;
            this->segmentCount = (this->arrayLength + this->segmentLength - 1) / this->segmentLength;
            this->segmentCount = this->segmentCount <= arity - 1 ? 1 : this->segmentCount - (arity - 1);
            this->arrayLength = (this->segmentCount + arity - 1) * this->segmentLength;
            this->segmentCountLength = this->segmentCount * this->segmentLength;
            fingerprints = new FingerprintType[arrayLength]();
            memset(fingerprints, 0, sizeof(FingerprintType) * arrayLength);
        }

        __device__ __host__ ~XorBinaryFuseFilter() {
            delete[] fingerprints;
            delete hasher;
        }

        __device__ __host__ bool Contain(const ItemType& item) const {
            uint64_t hash = (*hasher)(item + Seed);
            FingerprintType f = fingerprint(hash);
            for (int hi = 0; hi < 4; hi++) {
                size_t h = getHashFromHash(hash, hi);
                f ^= fingerprints[h];
            }
            return f == 0 ? true : false;
        }

        __device__ bool AllocateGPU(size_t arrayLength)
        {
            delete[] fingerprints;
            fingerprints = new FingerprintType[arrayLength];
            return true;

        }








        __host__ std::string Info() const {
            std::stringstream ss;
            ss << "4-wise XorBinaryFuseFilter Status:\n"
                << "\t\tKeys stored: " << Size() << "\n";
            return ss.str();
        }

        __device__ __host__ size_t Size() const { return size; }

        __device__ __host__ size_t SizeInBytes() const { return arrayLength * sizeof(FingerprintType); }


        // Function to load the filter from a file
        __host__ bool LoadFromFile(const std::string& filename) {
            std::ifstream in(filename, std::ios::binary);
            if (!in.is_open()) {
                return false;
            }
            in.read(reinterpret_cast<char*>(&size), sizeof(size));
            in.read(reinterpret_cast<char*>(&arrayLength), sizeof(arrayLength));
            in.read(reinterpret_cast<char*>(&segmentCount), sizeof(segmentCount));
            in.read(reinterpret_cast<char*>(&segmentCountLength), sizeof(segmentCountLength));
            in.read(reinterpret_cast<char*>(&segmentLength), sizeof(segmentLength));
            in.read(reinterpret_cast<char*>(&segmentLengthMask), sizeof(segmentLengthMask));
            delete[] fingerprints;
            fingerprints = new FingerprintType[arrayLength];
            in.read(reinterpret_cast<char*>(fingerprints), sizeof(FingerprintType) * arrayLength);
            in.close();
            return true;
        }
    };

} // namespace xorbinaryfusefilter_lowmem4wise

#endif  // FOURWISE_XOR_BINARY_FUSE_FILTER_XOR_FILTER_LOWMEM_H_
