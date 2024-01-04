#pragma once

#include "kernels/10_kernel_warptiling.cuh"
#include <algorithm>
#include <cassert>
//#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda_runtime.h>

__device__ __forceinline__ void ldgsts32Async(const uint32_t &smem_addr,
                                              const void *gmem_ptr) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, 4;\n"
               :
               : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void ldgsts_commit() {
  asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ uint32_t smemU32Addr(const void *smem_ptr) {
  uint32_t addr;
  asm("{.reg .u64 u64addr;\n"
      " cvta.to.shared.u64 u64addr, %1;\n"
      " cvt.u32.u64 %0, u64addr;}\n"
      : "=r"(addr)
      : "l"(smem_ptr));

  return addr;
}

__device__ __forceinline__ void lds128(float &reg0, float &reg1, float &reg2,
                                       float &reg3, const uint32_t &addr) {
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
               : "r"(addr));
}

__device__ __forceinline__ void stg128(const float &reg0, const float &reg1, const float &reg2,
                                       const float &reg3, void* gmemPtr) {
  asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};\n"
               :
               : "l"(gmemPtr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));
}

/*
   BM, BN, BK - thread block level  tile shape.
   WM, WN     - warp level tile shape.
   TM, TN     - thread level tile shape.
*/

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    runSgemmDoubleBuffering3(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  // Warps layout in thread block.
  // constexpr uint numWarpsM = BM / WM;
  constexpr uint numWarpsN = BN / WN;

  const uint warpId = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint laneId = threadIdx.x % WARPSIZE;

  // Warp tile position
  const uint warpIdX = warpId % numWarpsN;
  const uint warpIdY = warpId / numWarpsN;

  // Thread layout in  warp.
  constexpr uint warpDimX = WN / TN;
  constexpr uint warpDimY = WM / TM;
  static_assert(warpDimX * warpDimY == WARPSIZE);

  constexpr uint padAvoidBackConflict = 4;
  constexpr uint BMPadded = BM + padAvoidBackConflict;
  __shared__ float smem[2*(BMPadded + BN) * BK];
  float *aSmem = smem;
  float *bSmem = smem + 2 * BMPadded * BK;
  // __shared__ float aSmem[2][BMPadded * BK];
  // __shared__ float bSmem[2][BN * BK];

  float aFrag[2][TM];
  float bFrag[2][TN];
  float cFrag[TM][TN] = {0.0f};

  // Thread swizzled mapping.
  // The warp has a 2D layout: warpDimX x warpDimY. (mmaTidX, mmaTidY) is the
  // thread coordinate in the warp.
  // Use identity mapping for now.
  const uint mmaTidX = laneId % warpDimX;
  const uint mmaTidY = laneId / warpDimX;
  // const uint32_t mmaTidX = (lane_id / 2) % TN;
  // const uint32_t mmaTidY = (lane_id / TM) * 2 + (lane_id % 2);

  // Each thread load one element from A.
  constexpr uint numLoadAIters = BM * BK / NUM_THREADS;
  constexpr uint numLoadARowsPerIter = NUM_THREADS / BK;
  const uint loadAX = threadIdx.x % BK;
  const uint loadAY = threadIdx.x / BK;
  // Each thread load one element from B.
  constexpr uint numLoadBIters = BK * BN / NUM_THREADS;
  constexpr uint numLoadBRowsPerIter = NUM_THREADS / BN;
  const uint loadBX = threadIdx.x % BN;
  const uint loadBY = threadIdx.x / BN;

  uint32_t aStsAddr =
      smemU32Addr(aSmem + (threadIdx.x % BK) * BMPadded + (threadIdx.x / BK));
  // B_smem has very similar layout as B global memory.
  uint32_t bStsAddr = smemU32Addr(bSmem + threadIdx.x);

  // MOve global memory pointer to current tile.
  const uint cRow = blockIdx.y * BM;
  const uint cCol = blockIdx.x * BN;
  A += cRow * K;
  B += cCol;

  const char* aLdgPtr = (const char*)(A + loadAY * K + loadAX);
  const char* bLdgPtr = (const char*)(B + loadBY * N + loadBX);

  constexpr uint aSmemAddrShift =  BMPadded * BK * sizeof(float);
  constexpr uint bSmemAddrShift =  BN * BK * sizeof(float);
  constexpr uint aGmemAddrShift =  BK * sizeof(float);
  const uint bGmemAddrShift =  N * BK * sizeof(float);

  // Load first tile from global memory to shared memory.
  {
    // Load A from global memory to shared memory.
    #pragma unroll
    for (uint i = 0; i < numLoadAIters; ++i)
      ldgsts32Async(aStsAddr + i * numLoadARowsPerIter * sizeof(float),
                    aLdgPtr + i * numLoadARowsPerIter  * K * sizeof(float));

    // Load B from global memory to shared memory.
    #pragma unroll
    for (uint i = 0; i < numLoadBIters; ++i)
      ldgsts32Async(bStsAddr + i * numLoadBRowsPerIter * BN * sizeof(float),
                    bLdgPtr + i * numLoadBRowsPerIter * N * sizeof(float));

    ldgsts_commit();
    __syncthreads();

    // Advance A, B pointer to the next tile in K.
    aLdgPtr += aGmemAddrShift;
    bLdgPtr += bGmemAddrShift;

    // Update share memory store pointer to the other buffer.
    aStsAddr += aSmemAddrShift;
    bStsAddr += bSmemAddrShift;
  }

  // if (threadIdx.x == 0) {
  //   for (int i = 0; i < BK; ++i)
  //     printf("bsmem value %f\n", bSmem[i*BN]);
  // }

  uint aLdsAddr = smemU32Addr(aSmem + warpIdY * WM + mmaTidY * 4);
  uint bLdsAddr = smemU32Addr(bSmem + warpIdX * WN + mmaTidX * 4);

  // Load A first fragment. Assume no swizzling.
  #pragma unroll
  for (int i = 0; i < TM; i += 4)
    lds128(aFrag[0][i], aFrag[0][i + 1], aFrag[0][i + 2], aFrag[0][i + 3],
           aLdsAddr + i * warpDimY * sizeof(float));
  // Load B first fragment. Assume no swizzling.
  #pragma unroll
  for (int i = 0; i < TN; i += 4)
    lds128(bFrag[0][i], bFrag[0][i + 1], bFrag[0][i + 2], bFrag[0][i + 3],
           bLdsAddr + i * warpDimX * sizeof(float));

  const uint numKTiles = K / BK;
  uint bufferId = 0;
  uint nextBufferId = bufferId ^ 0x1;

  // K tiles loop.
  for (uint kTileId = 0; kTileId < numKTiles - 1; ++kTileId) {
    #pragma unroll
    for (uint k = 0; k < BK; ++k) {
      // uint bufferId = (k + 1) % 2;
      uint kNext = (k + 1) % BK;

      if (k == BK - 1) {
        ldgsts_commit();
        __syncthreads();

        // if (threadIdx.x == 0) {
        //   float *bptr = kTileId % 2 == 0 ? bSmem : bSmem + BK * BN;
        //   for (int i = 0; i < BK; ++i)
        //     printf("bsmem value %f\n", bptr[i*BN]);
        // }

        // Switch double buffer
        aLdsAddr = kTileId % 2 == 0 ? aLdsAddr + aSmemAddrShift : aLdsAddr - aSmemAddrShift;
        aStsAddr = kTileId % 2 == 0 ? aStsAddr - aSmemAddrShift : aStsAddr + aSmemAddrShift;
        bLdsAddr = kTileId % 2 == 0 ? bLdsAddr + bSmemAddrShift : bLdsAddr - bSmemAddrShift;
        bStsAddr = kTileId % 2 == 0 ? bStsAddr - bSmemAddrShift : bStsAddr + bSmemAddrShift;

        // Advance to global memory pointer to next tile.
        aLdgPtr += BK * sizeof(float);
        bLdgPtr += BK * N * sizeof(float);
        // A += BK * sizeof(float);
        // B += BK * N * sizeof(float);
      }

      // Load next A fragments from shared memory.
      #pragma unroll
      for (int i = 0; i < TM; i += 4)
        lds128(aFrag[nextBufferId][i], aFrag[nextBufferId][i + 1],
               aFrag[nextBufferId][i + 2], aFrag[nextBufferId][i + 3],
               aLdsAddr + (kNext * BMPadded + i * warpDimY) * sizeof(float));
      // Load next B fragments from shared memory.
      #pragma unroll
      for (int i = 0; i < TN; i += 4)
        lds128(bFrag[nextBufferId][i], bFrag[nextBufferId][i + 1],
               bFrag[nextBufferId][i + 2], bFrag[nextBufferId][i + 3],
               bLdsAddr + (kNext * BN + i * warpDimX) * sizeof(float));

      // Load next K tile from global memory to shared memory.
      if (k < numLoadAIters)
        ldgsts32Async(aStsAddr + k * numLoadARowsPerIter * sizeof(float),
                      aLdgPtr + k * numLoadARowsPerIter  * K * sizeof(float));
      if (k < numLoadBIters)
        ldgsts32Async(bStsAddr + k * numLoadBRowsPerIter * BN * sizeof(float),
                      bLdgPtr + k * numLoadBRowsPerIter * N * sizeof(float));

      // FFMA loop.
      #pragma unroll
      for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
          cFrag[i][j] += aFrag[bufferId][i] * bFrag[bufferId][j];
        } // end for j
      } // end for i
      // if (threadIdx.x == 0) {
      //   printf("kTile %d  k %d buffer id %d %d c41 %f a4k %f bk1 %f\n",
      //          kTileId, kTileId * BK + k, bufferId, nextBufferId,
      //          cFrag[4][1], aFrag[bufferId][4], bFrag[bufferId][1]);
      // }

      bufferId ^= 0x1;
      nextBufferId ^= 0x1;
    } // end for k
  }// end for KTileId

  // FFMA for the last tile.
  #pragma unroll
  for (int k = 0; k < BK; ++k) {
    uint bufferId = (k + 1) % 2;
    uint kNext = (k + 1) % BK;

    if (k < BK - 1) {
      // Load next A fragments from shared memory.
      #pragma unroll
      for (int i = 0; i < TM; i += 4)
        lds128(aFrag[bufferId][i], aFrag[bufferId][i + 1],
              aFrag[bufferId][i + 2], aFrag[bufferId][i + 3],
              aLdsAddr + (kNext * BMPadded + i * warpDimY) * sizeof(float));

      // Load next B fragments from shared memory.
      #pragma unroll
      for (int i = 0; i < TN; i += 4)
        lds128(bFrag[bufferId][i], bFrag[bufferId][i + 1],
               bFrag[bufferId][i + 2], bFrag[bufferId][i + 3],
               bLdsAddr + (kNext * BN + i * warpDimX) * sizeof(float));
    }// end if

    // FFMA loop
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
          cFrag[i][j] += aFrag[bufferId][i] * bFrag[bufferId][j];

    // if (threadIdx.x == 0) {
    //   printf("kTile %d  k %d buffer id %d %d c41 %f a4k %f bk1 %f\n",
    //           numKTiles-1, (numKTiles-1) * BK + k, bufferId, nextBufferId,
    //           cFrag[4][1], aFrag[bufferId][4], bFrag[bufferId][1]);
    // }
  } // end for k.


  // Move C_ptr to warp's output tile
  C += (cRow + warpIdY * WM) * N + cCol + warpIdX * WN;

  char* cLdgPtr = (char*)(C + mmaTidY * 4 * N + mmaTidX * 4);

  // Write result back to global memory.
  for (uint i = 0; i < TM; i+=4) {
    for (uint j = 0; j < TN; j+=4) {
      for (uint ii = 0; ii < 4; ++ii)
        stg128(cFrag[i+ii][j], cFrag[i+ii][j+1], cFrag[i+ii][j+2], cFrag[i+ii][j+3],
          cLdgPtr + ((i * warpDimY + ii) * N + j * warpDimX) * sizeof(float));
    }// end for j
  }// end for i
}
