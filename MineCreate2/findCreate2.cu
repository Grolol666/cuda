
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>


#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

#define TH_ELT(t, c0, c1, c2, c3, c4, d0, d1, d2, d3, d4)                                                              \
  {                                                                                                                    \
    t = ROTL64((d0 ^ d1 ^ d2 ^ d3 ^ d4), 1) ^ (c0 ^ c1 ^ c2 ^ c3 ^ c4);                                                \
  }

#define THETA(                                                                                                         \
  s00, s01, s02, s03, s04, s10, s11, s12, s13, s14, s20, s21, s22, s23, s24, s30, s31, s32, s33, s34, s40, s41, s42,   \
  s43, s44)                                                                                                            \
  {                                                                                                                    \
    TH_ELT(t0, s40, s41, s42, s43, s44, s10, s11, s12, s13, s14);                                                      \
    TH_ELT(t1, s00, s01, s02, s03, s04, s20, s21, s22, s23, s24);                                                      \
    TH_ELT(t2, s10, s11, s12, s13, s14, s30, s31, s32, s33, s34);                                                      \
    TH_ELT(t3, s20, s21, s22, s23, s24, s40, s41, s42, s43, s44);                                                      \
    TH_ELT(t4, s30, s31, s32, s33, s34, s00, s01, s02, s03, s04);                                                      \
    s00 ^= t0;                                                                                                         \
    s01 ^= t0;                                                                                                         \
    s02 ^= t0;                                                                                                         \
    s03 ^= t0;                                                                                                         \
    s04 ^= t0;                                                                                                         \
                                                                                                                       \
    s10 ^= t1;                                                                                                         \
    s11 ^= t1;                                                                                                         \
    s12 ^= t1;                                                                                                         \
    s13 ^= t1;                                                                                                         \
    s14 ^= t1;                                                                                                         \
                                                                                                                       \
    s20 ^= t2;                                                                                                         \
    s21 ^= t2;                                                                                                         \
    s22 ^= t2;                                                                                                         \
    s23 ^= t2;                                                                                                         \
    s24 ^= t2;                                                                                                         \
                                                                                                                       \
    s30 ^= t3;                                                                                                         \
    s31 ^= t3;                                                                                                         \
    s32 ^= t3;                                                                                                         \
    s33 ^= t3;                                                                                                         \
    s34 ^= t3;                                                                                                         \
                                                                                                                       \
    s40 ^= t4;                                                                                                         \
    s41 ^= t4;                                                                                                         \
    s42 ^= t4;                                                                                                         \
    s43 ^= t4;                                                                                                         \
    s44 ^= t4;                                                                                                         \
  }

#define RHOPI(                                                                                                         \
  s00, s01, s02, s03, s04, s10, s11, s12, s13, s14, s20, s21, s22, s23, s24, s30, s31, s32, s33, s34, s40, s41, s42,   \
  s43, s44)                                                                                                            \
  {                                                                                                                    \
    t0 = ROTL64(s10, (uint64_t)1);                                                                                     \
    s10 = ROTL64(s11, (uint64_t)44);                                                                                   \
    s11 = ROTL64(s41, (uint64_t)20);                                                                                   \
    s41 = ROTL64(s24, (uint64_t)61);                                                                                   \
    s24 = ROTL64(s42, (uint64_t)39);                                                                                   \
    s42 = ROTL64(s04, (uint64_t)18);                                                                                   \
    s04 = ROTL64(s20, (uint64_t)62);                                                                                   \
    s20 = ROTL64(s22, (uint64_t)43);                                                                                   \
    s22 = ROTL64(s32, (uint64_t)25);                                                                                   \
    s32 = ROTL64(s43, (uint64_t)8);                                                                                    \
    s43 = ROTL64(s34, (uint64_t)56);                                                                                   \
    s34 = ROTL64(s03, (uint64_t)41);                                                                                   \
    s03 = ROTL64(s40, (uint64_t)27);                                                                                   \
    s40 = ROTL64(s44, (uint64_t)14);                                                                                   \
    s44 = ROTL64(s14, (uint64_t)2);                                                                                    \
    s14 = ROTL64(s31, (uint64_t)55);                                                                                   \
    s31 = ROTL64(s13, (uint64_t)45);                                                                                   \
    s13 = ROTL64(s01, (uint64_t)36);                                                                                   \
    s01 = ROTL64(s30, (uint64_t)28);                                                                                   \
    s30 = ROTL64(s33, (uint64_t)21);                                                                                   \
    s33 = ROTL64(s23, (uint64_t)15);                                                                                   \
    s23 = ROTL64(s12, (uint64_t)10);                                                                                   \
    s12 = ROTL64(s21, (uint64_t)6);                                                                                    \
    s21 = ROTL64(s02, (uint64_t)3);                                                                                    \
    s02 = t0;                                                                                                          \
  }

#define KHI(                                                                                                           \
  s00, s01, s02, s03, s04, s10, s11, s12, s13, s14, s20, s21, s22, s23, s24, s30, s31, s32, s33, s34, s40, s41, s42,   \
  s43, s44)                                                                                                            \
  {                                                                                                                    \
    t0 = s00 ^ (~s10 & s20);                                                                                           \
    t1 = s10 ^ (~s20 & s30);                                                                                           \
    t2 = s20 ^ (~s30 & s40);                                                                                           \
    t3 = s30 ^ (~s40 & s00);                                                                                           \
    t4 = s40 ^ (~s00 & s10);                                                                                           \
    s00 = t0;                                                                                                          \
    s10 = t1;                                                                                                          \
    s20 = t2;                                                                                                          \
    s30 = t3;                                                                                                          \
    s40 = t4;                                                                                                          \
                                                                                                                       \
    t0 = s01 ^ (~s11 & s21);                                                                                           \
    t1 = s11 ^ (~s21 & s31);                                                                                           \
    t2 = s21 ^ (~s31 & s41);                                                                                           \
    t3 = s31 ^ (~s41 & s01);                                                                                           \
    t4 = s41 ^ (~s01 & s11);                                                                                           \
    s01 = t0;                                                                                                          \
    s11 = t1;                                                                                                          \
    s21 = t2;                                                                                                          \
    s31 = t3;                                                                                                          \
    s41 = t4;                                                                                                          \
                                                                                                                       \
    t0 = s02 ^ (~s12 & s22);                                                                                           \
    t1 = s12 ^ (~s22 & s32);                                                                                           \
    t2 = s22 ^ (~s32 & s42);                                                                                           \
    t3 = s32 ^ (~s42 & s02);                                                                                           \
    t4 = s42 ^ (~s02 & s12);                                                                                           \
    s02 = t0;                                                                                                          \
    s12 = t1;                                                                                                          \
    s22 = t2;                                                                                                          \
    s32 = t3;                                                                                                          \
    s42 = t4;                                                                                                          \
                                                                                                                       \
    t0 = s03 ^ (~s13 & s23);                                                                                           \
    t1 = s13 ^ (~s23 & s33);                                                                                           \
    t2 = s23 ^ (~s33 & s43);                                                                                           \
    t3 = s33 ^ (~s43 & s03);                                                                                           \
    t4 = s43 ^ (~s03 & s13);                                                                                           \
    s03 = t0;                                                                                                          \
    s13 = t1;                                                                                                          \
    s23 = t2;                                                                                                          \
    s33 = t3;                                                                                                          \
    s43 = t4;                                                                                                          \
                                                                                                                       \
    t0 = s04 ^ (~s14 & s24);                                                                                           \
    t1 = s14 ^ (~s24 & s34);                                                                                           \
    t2 = s24 ^ (~s34 & s44);                                                                                           \
    t3 = s34 ^ (~s44 & s04);                                                                                           \
    t4 = s44 ^ (~s04 & s14);                                                                                           \
    s04 = t0;                                                                                                          \
    s14 = t1;                                                                                                          \
    s24 = t2;                                                                                                          \
    s34 = t3;                                                                                                          \
    s44 = t4;                                                                                                          \
  }

#define IOTA(element, rc)                                                                                              \
  {                                                                                                                    \
    element ^= rc;                                                                                                     \
  }

__device__ const uint64_t RC[24] = { 0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
                                    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
                                    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
                                    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
                                    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
                                    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008 };

__device__ __forceinline__ void keccakf(uint64_t s[25])
{
    uint64_t t0, t1, t2, t3, t4;

    for (int i = 0; i < 24; i++) {
        THETA(
            s[0], s[5], s[10], s[15], s[20], s[1], s[6], s[11], s[16], s[21], s[2], s[7], s[12], s[17], s[22], s[3], s[8],
            s[13], s[18], s[23], s[4], s[9], s[14], s[19], s[24]);
        RHOPI(
            s[0], s[5], s[10], s[15], s[20], s[1], s[6], s[11], s[16], s[21], s[2], s[7], s[12], s[17], s[22], s[3], s[8],
            s[13], s[18], s[23], s[4], s[9], s[14], s[19], s[24]);
        KHI(
            s[0], s[5], s[10], s[15], s[20], s[1], s[6], s[11], s[16], s[21], s[2], s[7], s[12], s[17], s[22], s[3], s[8],
            s[13], s[18], s[23], s[4], s[9], s[14], s[19], s[24]);
        IOTA(s[0], RC[i]);
    }
}

#define N_BLOCK 1048576 //2^20
#define N_THREAD 512 //2^9
#define N_TOTAL (N_BLOCK * N_THREAD)

__device__ __forceinline__ void incUint256(uint64_t* a, const uint64_t val)
{
    uint64_t x = a[0] + val;
    if (x < a[0] || x < val)
    {
        uint64_t y = a[1] + 1;
        if (y < a[1])
        {
            uint64_t z = a[2] + 1;
            if (z < a[2])
            {
                a[3] = a[3] + 1;
            }
            a[2] = z;
        }
        a[1] = y;
    }
    a[0] = x;
}

__device__ __forceinline__ int compareAddr(uint8_t* a, uint8_t* b, uint32_t sizeToCompare)
{
    int k = 0;
    for (int i = 0; i < sizeToCompare; ++i)
    {
        if (a[i] != b[i]) 
            return k;
        ++k;
    }
    return k;
}

__global__ void find_keccak(uint8_t* addr, uint8_t* bytecodehash, uint8_t* addrToFind, const uint32_t sizeToCompare, uint8_t* saltFind)
{
    if (saltFind[32] != 0)
       return;

    //if (blockIdx.x == N_BLOCK - 1)
    //printf("find_keccak => blockDim.x => %u, blockIdx => %u, thread => %u %u\n", blockDim.x, blockIdx.x, threadIdx.x, sizeToCompare);

    __syncthreads();

    // prepare le salt
    uint8_t p_salt[32] = {};
    *(uint32_t*)p_salt = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint64_t* p_salt64 = (uint64_t*)p_salt;

    uint8_t input[136] = {};
    input[0] = 0xff;
    for (int i = 0; i < 20; ++i) {
        input[1 + i] = addr[i];
    }
    for (int i = 0; i < 32; ++i) {
        input[53 + i] = bytecodehash[i];
    }
    // pad 10*1
    input[85] = 1;
    // last bit
    input[135] |= 0x80;

    uint8_t result[20];
    uint64_t state[25] = {}; // Initialize with zeroes
    //uint64_t iter = 0;
    do
    {
        //++iter;

        for (int i = 0; i < 32; ++i) {
            input[21 + i] = p_salt[i];
        }
        for (int i = 0; i < 25; i++) {
            state[i] = (i < 17) ? *(uint64_t*)(input + i * 8) : 0;
        }
        keccakf(state);

        // prends que les 20 dernier octet
        uint8_t* p_state_u8 = (uint8_t*)state;
        for (int i = 0; i < 20; ++i)
            result[i] = p_state_u8[12 + i];


        if (compareAddr(result, addrToFind, sizeToCompare) == sizeToCompare) {
            //uint32_t v[32];
            for (int i = 0; i < 32; ++i) {
                saltFind[i] = input[21 + i];
                //uint8_t b = saltFind[i];
                //v[i] = b;
                //v[i] &= 0xFF;
            }

            // print de merde
            //for (int i = 0; i < 32; ++i) {
            //    printf("%.02x", v[i]);
            //}

            //printf("\n");
            for (int i = 0; i < 20; ++i) {
                saltFind[32 + i] = result[i];
                //uint8_t b = result[i];
                //v[i] = b;
                //v[i] &= 0xFF;
            }

            // print de merde
            //for (int i = 0; i < 20; ++i) {
            //    printf("%.02x", v[i]);
            //}
            //printf("\n");

            //find = true;
            //saltFind[32] = 1;
            //printf("This is a plain printf() call.\n");
            //printf("find_keccak find from => b:%u,t:%u on iter %llu\n", blockIdx.x, threadIdx.x, iter);
            return;
        }

        incUint256(p_salt64, N_TOTAL);

    } while (saltFind[32] == 0); // (saltFind[32] != 0);
    //printf("This is a plain printf() call.\n");
}

void uint8_to_hex_string(const uint8_t* values, int size)
{
    std::stringstream ss;
    for (int i = 0; i < size; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)values[i];
    }
    std::string hexString = ss.str();
    std::cout << hexString << std::endl;
}

uint8_t* cuda_init(uint8_t* pData, size_t s) {
    uint8_t* p_ADDR;
    cudaError_t cudaStatus = cudaMalloc(&p_ADDR, s);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error");
        return nullptr;
    }
    cudaStatus = cudaMemcpy(p_ADDR, pData, s, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy error");
        return nullptr;
    }
    return p_ADDR;
}

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

extern "C" cudaError_t find_keccak_salt(uint8_t * addr, uint8_t * bytecodehash, uint8_t * addrToFind, const uint32_t sizeToCmp)
{
    using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
    using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "install cuda...");
        return cudaStatus;
    }

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    uint8_t* p_ADDR = cuda_init(addr, 20);
    if (!p_ADDR) return cudaErrorMemoryAllocation;

    uint8_t* p_bytecodehash = cuda_init(bytecodehash, 32);
    if (!p_bytecodehash) return cudaErrorMemoryAllocation;

    uint8_t* p_addrToFind = cuda_init(addrToFind, sizeToCmp);
    if (!p_addrToFind) return cudaErrorMemoryAllocation;

    // prepare le resultat 
    uint8_t* p_result;
    cudaStatus = cudaMalloc(&p_result, 54);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    cudaStatus = cudaMemset(p_result, 0, 54);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    /*
    uint8_t* p_result_final;
    cudaStatus = cudaMalloc(&p_result_final, 4);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    cudaStatus = cudaMemset(p_result_final, 0, 4);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    */
    START_TIMER(find_keccak_timer);
    find_keccak <<<N_BLOCK, N_THREAD>>>(p_ADDR, p_bytecodehash, p_addrToFind, sizeToCmp, p_result);
    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("%s", cudaGetErrorString(cudaStatus));
    }
    

    uint8_t C[52];
    cudaStatus = cudaMemcpy(C, p_result, 52, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    END_TIMER(find_keccak_timer, "Keccak")

    uint8_to_hex_string((const uint8_t*)C, 32);
    uint8_to_hex_string((const uint8_t*)C + 32, 20);

    cudaStatus = cudaFree(p_ADDR);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    cudaStatus = cudaFree(p_bytecodehash);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    cudaStatus = cudaFree(p_addrToFind);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    cudaStatus = cudaFree(p_result);
    return cudaStatus;
}


//helper to new (malloc) uint8_t* //dont forget to delete [] after usage
uint8_t* hexToBytes(const std::string& hex)
{
    if (hex.size() < 2) return nullptr;
    std::string chex = (hex.substr(0, 2) == "0x") ? hex.substr(2) : hex;
    if (chex.empty() || (chex.size() & 1) == 1) return nullptr; //vide ou impair?

    uint8_t* pout = new uint8_t[chex.size() / 2];
    for (size_t i = 0; i < chex.size(); i += 2)
    {
        uint8_t a = std::stoi(chex.substr(i, 2), 0, 16);
        pout[i / 2] = a;
    }
    return pout;
}

int main(int argc, char* argv[])
{
    uint8_t* bytecodeHash = hexToBytes("0x690f45ae7b3433ad0918598d7d1cf6864ac095b0b374e85e7330e951280f328a");
    uint8_t* addr = hexToBytes("0x978BeCF0AEE83e6a2B93e48654250D106B6B7112");
    uint8_t* find_addr = hexToBytes("0xfee0000000");

    cudaError_t cudaStatus = find_keccak_salt(addr, bytecodeHash, find_addr, 3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ohhhh");
    }

    delete[] bytecodeHash;
    delete[] addr;
    delete[] find_addr;
}


//1048576 //(2^20)
/*
#define dtype uint32_t
#define SHIFTS (sizeof(dtype)*CHAR_BIT)
#define NIBBLES (SHIFTS/4)
#define ARRLEN 8

__device__ __forceinline__ void multiply(uint32_t *product, uint32_t *p_operand, uint32_t multiplier)
{
    int i;
    uint64_t partial = 0;
    for (i=0; i<ARRLEN; i++) {
        partial = partial + (uint64_t)multiplier * p_operand[i];
        product[i] = (dtype)partial;
        partial >>= SHIFTS;
    }
    //product[i] = (dtype)partial;
}
*/