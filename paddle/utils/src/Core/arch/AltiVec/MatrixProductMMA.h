// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Everton Constantino (everton.constantino@ibm.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
#define EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H

#pragma GCC target("cpu=power10")

#ifdef __has_builtin
#if !__has_builtin(__builtin_vsx_assemble_pair)
#define __builtin_vsx_assemble_pair __builtin_mma_assemble_pair
#endif
#endif

namespace Eigen {

namespace internal {

const static Packet16uc MMA_p16uc_SETCOMPLEX32_FIRST = {  0,  1,  2,  3,
                                                         16, 17, 18, 19,
                                                          4,  5,  6,  7,
                                                         20, 21, 22, 23};

const static Packet16uc MMA_p16uc_SETCOMPLEX32_SECOND = {  8,  9, 10, 11,
                                                          24, 25, 26, 27,
                                                          12, 13, 14, 15,
                                                          28, 29, 30, 31};
//[a,b],[ai,bi] = [a,ai] - This is equivalent to p16uc_GETREAL64
const static Packet16uc MMA_p16uc_SETCOMPLEX64_FIRST = {  0,  1,  2,  3,  4,  5,  6,  7,
                                                         16, 17, 18, 19, 20, 21, 22, 23};

//[a,b],[ai,bi] = [b,bi] - This is equivalent to p16uc_GETIMAG64
const static Packet16uc MMA_p16uc_SETCOMPLEX64_SECOND = {  8,  9, 10, 11, 12, 13, 14, 15,
                                                          24, 25, 26, 27, 28, 29, 30, 31};


// Grab two decouples real/imaginary PacketBlocks and return two coupled (real/imaginary pairs) PacketBlocks.
template<typename Packet, typename Packetc>
EIGEN_STRONG_INLINE void bcoupleMMA(PacketBlock<Packet,4>& taccReal, PacketBlock<Packet,4>& taccImag, PacketBlock<Packetc,8>& tRes, PacketBlock<Packetc, 4>& acc1, PacketBlock<Packetc, 4>& acc2)
{
  acc1.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], MMA_p16uc_SETCOMPLEX32_FIRST);
  acc1.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], MMA_p16uc_SETCOMPLEX32_FIRST);
  acc1.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], MMA_p16uc_SETCOMPLEX32_FIRST);
  acc1.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], MMA_p16uc_SETCOMPLEX32_FIRST);

  acc2.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], MMA_p16uc_SETCOMPLEX32_SECOND);
  acc2.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], MMA_p16uc_SETCOMPLEX32_SECOND);
  acc2.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], MMA_p16uc_SETCOMPLEX32_SECOND);
  acc2.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], MMA_p16uc_SETCOMPLEX32_SECOND);

  acc1.packet[0] = padd<Packetc>(tRes.packet[0], acc1.packet[0]);
  acc1.packet[1] = padd<Packetc>(tRes.packet[1], acc1.packet[1]);
  acc1.packet[2] = padd<Packetc>(tRes.packet[2], acc1.packet[2]);
  acc1.packet[3] = padd<Packetc>(tRes.packet[3], acc1.packet[3]);

  acc2.packet[0] = padd<Packetc>(tRes.packet[4], acc2.packet[0]);
  acc2.packet[1] = padd<Packetc>(tRes.packet[5], acc2.packet[1]);
  acc2.packet[2] = padd<Packetc>(tRes.packet[6], acc2.packet[2]);
  acc2.packet[3] = padd<Packetc>(tRes.packet[7], acc2.packet[3]);
}

template<>
EIGEN_STRONG_INLINE void bcoupleMMA<Packet2d, Packet1cd>(PacketBlock<Packet2d,4>& taccReal, PacketBlock<Packet2d,4>& taccImag, PacketBlock<Packet1cd,8>& tRes, PacketBlock<Packet1cd, 4>& acc1, PacketBlock<Packet1cd, 4>& acc2)
{
  acc1.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], MMA_p16uc_SETCOMPLEX64_FIRST);
  acc1.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], MMA_p16uc_SETCOMPLEX64_FIRST);
  acc1.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], MMA_p16uc_SETCOMPLEX64_FIRST);
  acc1.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], MMA_p16uc_SETCOMPLEX64_FIRST);

  acc2.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], MMA_p16uc_SETCOMPLEX64_SECOND);
  acc2.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], MMA_p16uc_SETCOMPLEX64_SECOND);
  acc2.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], MMA_p16uc_SETCOMPLEX64_SECOND);
  acc2.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], MMA_p16uc_SETCOMPLEX64_SECOND);

  acc1.packet[0] = padd<Packet1cd>(tRes.packet[0], acc1.packet[0]);
  acc1.packet[1] = padd<Packet1cd>(tRes.packet[1], acc1.packet[1]);
  acc1.packet[2] = padd<Packet1cd>(tRes.packet[2], acc1.packet[2]);
  acc1.packet[3] = padd<Packet1cd>(tRes.packet[3], acc1.packet[3]);

  acc2.packet[0] = padd<Packet1cd>(tRes.packet[4], acc2.packet[0]);
  acc2.packet[1] = padd<Packet1cd>(tRes.packet[5], acc2.packet[1]);
  acc2.packet[2] = padd<Packet1cd>(tRes.packet[6], acc2.packet[2]);
  acc2.packet[3] = padd<Packet1cd>(tRes.packet[7], acc2.packet[3]);
}

template<typename Scalar, typename Packet>
EIGEN_STRONG_INLINE Packet ploadLhsMMA(const Scalar *lhs)
{
  return *((Packet *)lhs);
}

template<typename Packet>
EIGEN_STRONG_INLINE PacketBlock<Packet,2> pmul(const PacketBlock<Packet,2>& a, const Packet& b)
{
  PacketBlock<Packet,2> pb;
  pb.packet[0] = a.packet[0]*b;
  pb.packet[1] = a.packet[1]*b;
  return pb;
}

template<typename Scalar, typename Packet>
EIGEN_STRONG_INLINE void bsetzeroMMA(__vector_quad *acc)
{
  __builtin_mma_xxsetaccz(acc);
}

template<typename DataMapper, typename Index, typename Packet>
EIGEN_STRONG_INLINE void storeAccumulator(Index i, Index j, const DataMapper& data, const Packet& alpha, __vector_quad *acc)
{
  PacketBlock<Packet, 4> result;
  __builtin_mma_disassemble_acc(&result.packet, acc);

  result.packet[0] = pmadd<Packet>(alpha, result.packet[0], data.template loadPacket<Packet>(i, j + 0));
  result.packet[1] = pmadd<Packet>(alpha, result.packet[1], data.template loadPacket<Packet>(i, j + 1));
  result.packet[2] = pmadd<Packet>(alpha, result.packet[2], data.template loadPacket<Packet>(i, j + 2));
  result.packet[3] = pmadd<Packet>(alpha, result.packet[3], data.template loadPacket<Packet>(i, j + 3));

  data.template storePacketBlock<Packet, 4>(i, j, result);
}

template<typename DataMapper, typename Index, typename Packet, typename Packetc, int N>
EIGEN_STRONG_INLINE void storeComplexAccumulator(Index i, Index j, const DataMapper& data, const Packet& alphaReal, const Packet& alphaImag, __vector_quad *accReal, __vector_quad *accImag, const int accColsC)
{
  PacketBlock<Packet, 4> resultReal, resultImag;
  __builtin_mma_disassemble_acc(&resultReal.packet, accReal);
  __builtin_mma_disassemble_acc(&resultImag.packet, accImag);

  PacketBlock<Packet,4> taccReal, taccImag;
  taccReal.packet[0] = pmul<Packet>(resultReal.packet[0], alphaReal);
  taccReal.packet[1] = pmul<Packet>(resultReal.packet[1], alphaReal);
  taccReal.packet[2] = pmul<Packet>(resultReal.packet[2], alphaReal);
  taccReal.packet[3] = pmul<Packet>(resultReal.packet[3], alphaReal);

  taccImag.packet[0] = pmul<Packet>(resultImag.packet[0], alphaReal);
  taccImag.packet[1] = pmul<Packet>(resultImag.packet[1], alphaReal);
  taccImag.packet[2] = pmul<Packet>(resultImag.packet[2], alphaReal);
  taccImag.packet[3] = pmul<Packet>(resultImag.packet[3], alphaReal);

  taccReal.packet[0] = psub<Packet>(taccReal.packet[0], pmul<Packet>(resultImag.packet[0], alphaImag));
  taccReal.packet[1] = psub<Packet>(taccReal.packet[1], pmul<Packet>(resultImag.packet[1], alphaImag));
  taccReal.packet[2] = psub<Packet>(taccReal.packet[2], pmul<Packet>(resultImag.packet[2], alphaImag));
  taccReal.packet[3] = psub<Packet>(taccReal.packet[3], pmul<Packet>(resultImag.packet[3], alphaImag));

  taccImag.packet[0] = pmadd<Packet>(resultReal.packet[0], alphaImag, taccImag.packet[0]);
  taccImag.packet[1] = pmadd<Packet>(resultReal.packet[1], alphaImag, taccImag.packet[1]);
  taccImag.packet[2] = pmadd<Packet>(resultReal.packet[2], alphaImag, taccImag.packet[2]);
  taccImag.packet[3] = pmadd<Packet>(resultReal.packet[3], alphaImag, taccImag.packet[3]);

  PacketBlock<Packetc, 8> tRes;
  tRes.packet[0] = data.template loadPacket<Packetc>(i + N*accColsC, j + 0);
  tRes.packet[1] = data.template loadPacket<Packetc>(i + N*accColsC, j + 1);
  tRes.packet[2] = data.template loadPacket<Packetc>(i + N*accColsC, j + 2);
  tRes.packet[3] = data.template loadPacket<Packetc>(i + N*accColsC, j + 3);

  tRes.packet[4] = data.template loadPacket<Packetc>(i + (N+1)*accColsC, j + 0);
  tRes.packet[5] = data.template loadPacket<Packetc>(i + (N+1)*accColsC, j + 1);
  tRes.packet[6] = data.template loadPacket<Packetc>(i + (N+1)*accColsC, j + 2);
  tRes.packet[7] = data.template loadPacket<Packetc>(i + (N+1)*accColsC, j + 3);

  PacketBlock<Packetc, 4> acc1, acc2;
  bcoupleMMA<Packet, Packetc>(taccReal, taccImag, tRes, acc1, acc2);

  data.template storePacketBlock<Packetc, 4>(i + N*accColsC, j, acc1);
  data.template storePacketBlock<Packetc, 4>(i + (N+1)*accColsC, j, acc2);
}

// Defaults to float32, since Eigen still supports C++03 we can't use default template arguments
template<typename LhsPacket, typename RhsPacket, bool NegativeAccumulate>
EIGEN_STRONG_INLINE void pgerMMA(__vector_quad *acc, const RhsPacket& a, const LhsPacket& b)
{
  if(NegativeAccumulate)
  {
    __builtin_mma_xvf32gernp(acc, (__vector unsigned char)a, (__vector unsigned char)b);
  } else {
    __builtin_mma_xvf32gerpp(acc, (__vector unsigned char)a, (__vector unsigned char)b);
  }
}

template<>
EIGEN_STRONG_INLINE void pgerMMA<Packet2d, PacketBlock<Packet2d, 2>, false>(__vector_quad *acc, const PacketBlock<Packet2d,2>& a, const Packet2d& b)
{
  __vector_pair *a0 = (__vector_pair *)(&a.packet[0]);
  __builtin_mma_xvf64gerpp(acc, *a0, (__vector unsigned char)b);
}

template<>
EIGEN_STRONG_INLINE void pgerMMA<Packet2d, PacketBlock<Packet2d, 2>, true>(__vector_quad *acc, const PacketBlock<Packet2d, 2>& a, const Packet2d& b)
{
  __vector_pair *a0 = (__vector_pair *)(&a.packet[0]);
  __builtin_mma_xvf64gernp(acc, *a0, (__vector unsigned char)b);
}

template<>
EIGEN_STRONG_INLINE void pgerMMA<Packet2d, __vector_pair, false>(__vector_quad *acc, const __vector_pair& a, const Packet2d& b)
{
  __builtin_mma_xvf64gerpp(acc, (__vector_pair)a, (__vector unsigned char)b);
}

template<>
EIGEN_STRONG_INLINE void pgerMMA<Packet2d, __vector_pair, true>(__vector_quad *acc, const __vector_pair& a, const Packet2d& b)
{
  __builtin_mma_xvf64gernp(acc, (__vector_pair)a, (__vector unsigned char)b);
}

template<>
EIGEN_STRONG_INLINE void pgerMMA<Packet4f, __vector_pair, false>(__vector_quad *acc, const __vector_pair& a, const Packet4f& b)
{
  // Just for compilation
  EIGEN_UNUSED_VARIABLE(acc)
  EIGEN_UNUSED_VARIABLE(a)
  EIGEN_UNUSED_VARIABLE(b)
}

template<>
EIGEN_STRONG_INLINE void pgerMMA<Packet4f, __vector_pair, true>(__vector_quad *acc, const __vector_pair& a, const Packet4f& b)
{
  // Just for compilation
  EIGEN_UNUSED_VARIABLE(acc)
  EIGEN_UNUSED_VARIABLE(a)
  EIGEN_UNUSED_VARIABLE(b)
}

// This is necessary because ploadRhs for double returns a pair of vectors when MMA is enabled.
template<typename Scalar, typename Packet>
EIGEN_STRONG_INLINE void ploadRhsMMA(const Scalar *rhs, Packet &rhsV)
{
  rhsV = *((Packet *)rhs);
} 

template<>
EIGEN_STRONG_INLINE void ploadRhsMMA<double, PacketBlock<Packet2d, 2> >(const double *rhs, PacketBlock<Packet2d, 2> &rhsV)
{
  rhsV.packet[0] = *((Packet2d *)rhs      );
  rhsV.packet[1] = *(((Packet2d *)rhs) + 1);
}

template<>
EIGEN_STRONG_INLINE void ploadRhsMMA<double, __vector_pair>(const double *rhs, __vector_pair &rhsV)
{
  __builtin_vsx_assemble_pair(&rhsV, (__vector unsigned char)(*(((Packet2d *)rhs) + 1)), (__vector unsigned char)(*((Packet2d *)rhs)));
}

#define MICRO_MMA_DST \
  __vector_quad *accZero0, __vector_quad *accZero1, __vector_quad *accZero2, \
  __vector_quad *accZero3, __vector_quad *accZero4, __vector_quad *accZero5, \
  __vector_quad *accZero6, __vector_quad *accZero7

#define MICRO_MMA_SRC \
  const Scalar **lhs_ptr0, const Scalar **lhs_ptr1, const Scalar **lhs_ptr2, \
  const Scalar **lhs_ptr3, const Scalar **lhs_ptr4, const Scalar **lhs_ptr5, \
  const Scalar **lhs_ptr6, const Scalar **lhs_ptr7

#define MICRO_MMA_ONE \
  if (sizeof(Scalar) == sizeof(float)) { \
    MICRO_MMA<unroll_factor, Scalar, Packet, RhsPacket, accRows, accCols>(\
      &lhs_ptr0, &lhs_ptr1, &lhs_ptr2, &lhs_ptr3, &lhs_ptr4, &lhs_ptr5, &lhs_ptr6, &lhs_ptr7, \
      rhs_ptr, \
      &accZero0, &accZero1, &accZero2, &accZero3, &accZero4, &accZero5, &accZero6, &accZero7); \
  } else { \
    MICRO_MMA<unroll_factor, Scalar, Packet, __vector_pair, accRows, accCols>(\
      &lhs_ptr0, &lhs_ptr1, &lhs_ptr2, &lhs_ptr3, &lhs_ptr4, &lhs_ptr5, &lhs_ptr6, &lhs_ptr7, \
      rhs_ptr, \
      &accZero0, &accZero1, &accZero2, &accZero3, &accZero4, &accZero5, &accZero6, &accZero7); \
  }

#define MICRO_MMA_WORK_ONE(iter) \
  if (N > iter) { \
    Packet lhsV = ploadLhsMMA<Scalar, Packet>(*lhs_ptr##iter); \
    pgerMMA<Packet, RhsPacket, false>(accZero##iter, rhsV, lhsV); \
    *lhs_ptr##iter += accCols; \
  } else { \
    EIGEN_UNUSED_VARIABLE(accZero##iter); \
    EIGEN_UNUSED_VARIABLE(lhs_ptr##iter); \
  }

#define MICRO_MMA_UNROLL(func) \
  func(0) func(1) func(2) func(3) func(4) func(5) func(6) func(7)

#define MICRO_MMA_WORK MICRO_MMA_UNROLL(MICRO_MMA_WORK_ONE)

#define MICRO_MMA_DST_PTR_ONE(iter) \
  if (unroll_factor > iter){ \
    bsetzeroMMA<Scalar, Packet>(&accZero##iter); \
  } else { \
    EIGEN_UNUSED_VARIABLE(accZero##iter); \
  }

#define MICRO_MMA_DST_PTR MICRO_MMA_UNROLL(MICRO_MMA_DST_PTR_ONE)

#define MICRO_MMA_SRC_PTR_ONE(iter) \
  if (unroll_factor > iter) { \
    lhs_ptr##iter = lhs_base + ( (row/accCols) + iter )*strideA*accCols + accCols*offsetA; \
  } else { \
    EIGEN_UNUSED_VARIABLE(lhs_ptr##iter); \
  }

#define MICRO_MMA_SRC_PTR MICRO_MMA_UNROLL(MICRO_MMA_SRC_PTR_ONE)

#define MICRO_MMA_PREFETCH_ONE(iter) \
  if (unroll_factor > iter){ \
    prefetch(lhs_ptr##iter); \
  }

#define MICRO_MMA_PREFETCH MICRO_MMA_UNROLL(MICRO_MMA_PREFETCH_ONE)

#define MICRO_MMA_STORE_ONE(iter) \
  if (unroll_factor > iter){ \
    storeAccumulator<DataMapper, Index, Packet>(row + iter*accCols, col, res, pAlpha, &accZero##iter); \
  }

#define MICRO_MMA_STORE MICRO_MMA_UNROLL(MICRO_MMA_STORE_ONE)

// PEEL_MMA loop factor.
#define PEEL_MMA 10

template<int N, typename Scalar, typename Packet, typename RhsPacket, const Index accRows, const Index accCols>
EIGEN_STRONG_INLINE void MICRO_MMA(
  MICRO_MMA_SRC,
  const Scalar* &rhs_ptr,
  MICRO_MMA_DST)
  {
    RhsPacket rhsV;
    ploadRhsMMA<Scalar, RhsPacket>(rhs_ptr, rhsV);
    MICRO_MMA_WORK
    rhs_ptr += accRows;
  }

template<int unroll_factor, typename Scalar, typename Packet, typename RhsPacket, typename DataMapper, typename Index, const Index accRows, const Index accCols>
EIGEN_STRONG_INLINE void gemm_unrolled_MMA_iteration(
  const DataMapper& res,
  const Scalar *lhs_base,
  const Scalar *rhs_base,
  Index depth,
  Index strideA,
  Index offsetA,
  Index& row,
  Index col,
  const Packet& pAlpha)
{
  const Scalar *rhs_ptr = rhs_base;
  const Scalar *lhs_ptr0, *lhs_ptr1, *lhs_ptr2, *lhs_ptr3, *lhs_ptr4, *lhs_ptr5, *lhs_ptr6, *lhs_ptr7;
  __vector_quad accZero0, accZero1, accZero2, accZero3, accZero4, accZero5, accZero6, accZero7;

  asm("#unrolled MMA start");
  MICRO_MMA_SRC_PTR
  MICRO_MMA_DST_PTR

  Index k = 0;
  for(; k + PEEL_MMA <= depth; k+= PEEL_MMA)
  {
    prefetch(rhs_ptr);
    MICRO_MMA_PREFETCH
    for (int l = 0; l < PEEL_MMA; l++) {
      MICRO_MMA_ONE
    }
  }
  for(; k < depth; k++)
  {
    MICRO_MMA_ONE
  }
  MICRO_MMA_STORE

  row += unroll_factor*accCols;
  asm("#unrolled MMA end");
}

template<typename Scalar, typename Index, typename Packet, typename RhsPacket, typename DataMapper, const Index accRows, const Index accCols>
void gemmMMA(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index rows, Index depth, Index cols, Scalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB)
{
      const Index remaining_rows = rows % accCols;
      const Index remaining_cols = cols % accRows;

      if( strideA == -1 ) strideA = depth;
      if( strideB == -1 ) strideB = depth;

      const Packet pAlpha = pset1<Packet>(alpha);
      const Packet pMask  = bmask<Packet>((const int)(remaining_rows));

      Index col = 0;
      for(; col + accRows <= cols; col += accRows)
      {
        const Scalar *rhs_base = blockB + col*strideB + accRows*offsetB;
        const Scalar *lhs_base = blockA;

        Index row = 0;
#define MAX_MMA_UNROLL 7
        while(row + MAX_MMA_UNROLL*accCols <= rows){
          gemm_unrolled_MMA_iteration<MAX_MMA_UNROLL, Scalar, Packet, RhsPacket, DataMapper, Index, accRows, accCols>(res, lhs_base, rhs_base, depth, strideA, offsetA, row, col, pAlpha);
        }
        switch( (rows-row)/accCols ){
#if MAX_MMA_UNROLL > 7
          case 7:
            gemm_unrolled_MMA_iteration<7, Scalar, Packet, RhsPacket, DataMapper, Index, accRows, accCols>(res, lhs_base, rhs_base, depth, strideA, offsetA, row, col, pAlpha);
            break;
#endif
#if MAX_MMA_UNROLL > 6
          case 6:
            gemm_unrolled_MMA_iteration<6, Scalar, Packet, RhsPacket, DataMapper, Index, accRows, accCols>(res, lhs_base, rhs_base, depth, strideA, offsetA, row, col, pAlpha);
            break;
#endif
#if MAX_MMA_UNROLL > 5
          case 5:
            gemm_unrolled_MMA_iteration<5, Scalar, Packet, RhsPacket, DataMapper, Index, accRows, accCols>(res, lhs_base, rhs_base, depth, strideA, offsetA, row, col, pAlpha);
            break;
#endif
#if MAX_MMA_UNROLL > 4
          case 4:
            gemm_unrolled_MMA_iteration<4, Scalar, Packet, RhsPacket, DataMapper, Index, accRows, accCols>(res, lhs_base, rhs_base, depth, strideA, offsetA, row, col, pAlpha);
            break;
#endif
#if MAX_MMA_UNROLL > 3
          case 3:
            gemm_unrolled_MMA_iteration<3, Scalar, Packet, RhsPacket, DataMapper, Index, accRows, accCols>(res, lhs_base, rhs_base, depth, strideA, offsetA, row, col, pAlpha);
            break;
#endif
#if MAX_MMA_UNROLL > 2
          case 2:
            gemm_unrolled_MMA_iteration<2, Scalar, Packet, RhsPacket, DataMapper, Index, accRows, accCols>(res, lhs_base, rhs_base, depth, strideA, offsetA, row, col, pAlpha);
            break;
#endif
#if MAX_MMA_UNROLL > 1
          case 1:
            gemm_unrolled_MMA_iteration<1, Scalar, Packet, RhsPacket, DataMapper, Index, accRows, accCols>(res, lhs_base, rhs_base, depth, strideA, offsetA, row, col, pAlpha);
            break;
#endif
          default:
            break;
        }
#undef MAX_MMA_UNROLL

        if(remaining_rows > 0)
        {
          gemm_extra_row<Scalar, Packet, DataMapper, Index, accRows>(res, lhs_base, rhs_base, depth, strideA, offsetA, row, col, cols, remaining_rows, pAlpha, pMask);
        }
    }

    if(remaining_cols > 0)
    {
      const Scalar *rhs_base = blockB + col*strideB + remaining_cols*offsetB;
      const Scalar *lhs_base = blockA;

      for(; col < cols; col++)
      {
        Index row = 0;

        gemm_unrolled_col<Scalar, Packet, DataMapper, Index, accCols>(res, lhs_base, rhs_base, depth, strideA, offsetA, row, rows, col, remaining_cols, pAlpha);

        if (remaining_rows > 0)
        {
          gemm_extra_col<Scalar, Packet, DataMapper, Index, accRows>(res, lhs_base, rhs_base, depth, strideA, offsetA, row, col, remaining_rows, remaining_cols, pAlpha);
        }
        rhs_base++;
      }
    }
}

template<typename LhsScalar, typename RhsScalar, typename Scalarc, typename Scalar, typename Index, typename Packet, typename Packetc, typename RhsPacket, typename DataMapper, const int accRows, const int accCols, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
void gemm_complexMMA(const DataMapper& res, const LhsScalar* blockAc, const RhsScalar* blockBc,
          Index rows, Index depth, Index cols, Scalarc alpha, Index strideA, Index strideB, Index offsetA, Index offsetB)
{
      const int remaining_rows = rows % accCols;
      const int remaining_cols = cols % accRows;
      const int accColsC = accCols / 2;
      int advanceCols = 2;
      int advanceRows = 2;

      if(LhsIsReal) advanceRows = 1;
      if(RhsIsReal) advanceCols = 1;

      if( strideA == -1 ) strideA = depth;
      if( strideB == -1 ) strideB = depth;

      const Packet pAlphaReal = pset1<Packet>(alpha.real());
      const Packet pAlphaImag = pset1<Packet>(alpha.imag());

      const Scalar *blockA = (Scalar *) blockAc;
      const Scalar *blockB = (Scalar *) blockBc;

      Packet conj = pset1<Packet>((Scalar)-1.0f);

      Index col = 0;
      for(; col + accRows <= cols; col += accRows)
      {
        const Scalar *rhs_base = blockB + ( (advanceCols*col)/accRows     )*strideB*accRows;
        const Scalar *lhs_base = blockA;

        Index row = 0;

        for(; row + accCols <= rows; row += accCols)
        {
          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *rhs_ptr_imag = rhs_ptr + accRows*strideB;
          const Scalar *lhs_ptr = lhs_base + ((advanceRows*row)/accCols)*strideA*accCols;
          const Scalar *lhs_ptr_imag = lhs_ptr + accCols*strideA;

          __vector_quad accReal, accImag;
          __builtin_mma_xxsetaccz(&accReal);
          __builtin_mma_xxsetaccz(&accImag);

          lhs_ptr += accCols*offsetA;
          if(!LhsIsReal)
            lhs_ptr_imag += accCols*offsetA;
          rhs_ptr += accRows*offsetB;
          if(!RhsIsReal)
            rhs_ptr_imag += accRows*offsetB;
          for(Index k = 0; k < depth; k++)
          {
            Packet lhsV = ploadLhsMMA<Scalar, Packet>(lhs_ptr);
            RhsPacket rhsV = ploadRhs<Scalar, RhsPacket>(rhs_ptr);

            Packet lhsVi = ploadLhsMMA<Scalar, Packet>(lhs_ptr_imag);
            RhsPacket rhsVi = ploadRhs<Scalar, RhsPacket>(rhs_ptr_imag);

            if(ConjugateLhs && !LhsIsReal) lhsVi = pmul<Packet>(lhsVi, conj);
            if(ConjugateRhs && !RhsIsReal) rhsVi = pmul<Packet>(rhsVi, conj);

            if(LhsIsReal)
            {
              pgerMMA<Packet, RhsPacket, false>(&accReal,  rhsV,  lhsV);
              pgerMMA<Packet, RhsPacket, false>(&accImag, rhsVi,  lhsV);
            } else if(RhsIsReal) {
              pgerMMA<Packet, RhsPacket, false>(&accReal,  rhsV,  lhsV);
              pgerMMA<Packet, RhsPacket, false>(&accImag,  rhsV, lhsVi);
            } else {
              pgerMMA<Packet, RhsPacket, false>(&accReal,  rhsV,  lhsV);
              pgerMMA<Packet, RhsPacket,  true>(&accReal, rhsVi, lhsVi);
              pgerMMA<Packet, RhsPacket, false>(&accImag, rhsVi,  lhsV);
              pgerMMA<Packet, RhsPacket, false>(&accImag,  rhsV, lhsVi);
            }

            lhs_ptr += accCols;
            rhs_ptr += accRows;
            if(!LhsIsReal)
              lhs_ptr_imag += accCols;
            if(!RhsIsReal)
              rhs_ptr_imag += accRows;
          }

          storeComplexAccumulator<DataMapper, Index, Packet, Packetc, 0>(row, col, res, pAlphaReal, pAlphaImag, &accReal, &accImag, accColsC);
        }

          if(remaining_rows > 0)
          {
            const Scalar *rhs_ptr  = rhs_base;
            const Scalar *rhs_ptr_imag = rhs_ptr + accRows*strideB;
            const Scalar *lhs_ptr = lhs_base + ((advanceRows*row)/accCols)*strideA*accCols;
            const Scalar *lhs_ptr_imag = lhs_ptr + remaining_rows*strideA;

            lhs_ptr += remaining_rows*offsetA;
            if(!LhsIsReal)
              lhs_ptr_imag += remaining_rows*offsetA;
            rhs_ptr += accRows*offsetB;
            if(!RhsIsReal)
              rhs_ptr_imag += accRows*offsetB;
            for(Index k = 0; k < depth; k++)
            {
              for(Index arow = 0; arow < remaining_rows; arow++)
              {
                Scalar lhs_real = lhs_ptr[arow];
                Scalar lhs_imag;
                if(!LhsIsReal) lhs_imag = lhs_ptr_imag[arow];

                Scalarc lhsc;

                lhsc.real(lhs_real);
                if(!LhsIsReal)
                {
                  if(ConjugateLhs) 
                    lhsc.imag(-lhs_imag);
                  else
                    lhsc.imag(lhs_imag);
                } else {
                  //Lazy approach for now
                  lhsc.imag((Scalar)0);
                }

                for(int acol = 0; acol < accRows; acol++ )
                {
                  Scalar rhs_real = rhs_ptr[acol];
                  Scalar rhs_imag;
                  if(!RhsIsReal) rhs_imag = rhs_ptr_imag[acol];
                  Scalarc rhsc;

                  rhsc.real(rhs_real);
                  if(!RhsIsReal)
                  {
                    if(ConjugateRhs)
                      rhsc.imag(-rhs_imag);
                    else
                      rhsc.imag(rhs_imag);
                  } else {
                    //Lazy approach for now
                    rhsc.imag((Scalar)0);
                  }
                  res(row + arow, col + acol) += alpha*lhsc*rhsc;
                }
              }
              rhs_ptr += accRows;
              lhs_ptr += remaining_rows;
              if(!LhsIsReal)
                lhs_ptr_imag += remaining_rows;
              if(!RhsIsReal)
                rhs_ptr_imag += accRows;
            }
          }
      }

      if(remaining_cols > 0)
      {
        const Scalar *rhs_base = blockB + ( (advanceCols*col)/accRows     )*strideB*accRows;
        const Scalar *lhs_base = blockA;
        Index row = 0;

        for(; row + accCols <= rows; row += accCols)
        {
          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *rhs_ptr_imag = rhs_ptr + remaining_cols*strideB;
          const Scalar *lhs_ptr = lhs_base + ((advanceRows*row)/accCols)*strideA*accCols;
          const Scalar *lhs_ptr_imag = lhs_ptr + accCols*strideA;

          lhs_ptr += accCols*offsetA;
          if(!LhsIsReal)
            lhs_ptr_imag += accCols*offsetA;
          rhs_ptr += remaining_cols*offsetB;
          if(!RhsIsReal)
            rhs_ptr_imag += remaining_cols*offsetB;
          Scalarc scalarAcc[4][4];
          for(Index arow = 0; arow < 4; arow++ )
          {
            for(Index acol = 0; acol < 4; acol++ )
            {
              scalarAcc[arow][acol].real((Scalar)0.0f);
              scalarAcc[arow][acol].imag((Scalar)0.0f);
            }
          }
          for(Index k = 0; k < depth; k++)
          {
            for(Index arow = 0; arow < accCols; arow++)
            {
              Scalar lhs_real = lhs_ptr[arow];
              Scalar lhs_imag;
              if(!LhsIsReal) 
              {
                lhs_imag = lhs_ptr_imag[arow];

                if(ConjugateLhs)
                  lhs_imag *= -1;
              } else {
                lhs_imag = (Scalar)0;
              }

              for(int acol = 0; acol < remaining_cols; acol++ )
              {
                Scalar rhs_real = rhs_ptr[acol];
                Scalar rhs_imag;
                if(!RhsIsReal)
                {
                  rhs_imag = rhs_ptr_imag[acol];

                  if(ConjugateRhs)
                    rhs_imag *= -1;
                } else {
                  rhs_imag = (Scalar)0;
                }

                scalarAcc[arow][acol].real(scalarAcc[arow][acol].real() + lhs_real*rhs_real - lhs_imag*rhs_imag);
                scalarAcc[arow][acol].imag(scalarAcc[arow][acol].imag() + lhs_imag*rhs_real + lhs_real*rhs_imag);
              }
            }
            rhs_ptr += remaining_cols;
            lhs_ptr += accCols;
            if(!RhsIsReal)
              rhs_ptr_imag += remaining_cols;
            if(!LhsIsReal)
              lhs_ptr_imag += accCols;
          }
          for(int arow = 0; arow < accCols; arow++ )
          {
            for(int acol = 0; acol < remaining_cols; acol++ )
            {
              Scalar accR = scalarAcc[arow][acol].real();
              Scalar accI = scalarAcc[arow][acol].imag();
              Scalar aR = alpha.real();
              Scalar aI = alpha.imag();
              Scalar resR = res(row + arow, col + acol).real();
              Scalar resI = res(row + arow, col + acol).imag();

              res(row + arow, col + acol).real(resR + accR*aR - accI*aI);
              res(row + arow, col + acol).imag(resI + accR*aI + accI*aR);
            }
          }
        }

        if(remaining_rows > 0)
        {
          const Scalar *rhs_ptr  = rhs_base;
          const Scalar *rhs_ptr_imag = rhs_ptr + remaining_cols*strideB;
          const Scalar *lhs_ptr = lhs_base + ((advanceRows*row)/accCols)*strideA*accCols;
          const Scalar *lhs_ptr_imag = lhs_ptr + remaining_rows*strideA;

          lhs_ptr += remaining_rows*offsetA;
          if(!LhsIsReal)
            lhs_ptr_imag += remaining_rows*offsetA;
          rhs_ptr += remaining_cols*offsetB;
          if(!RhsIsReal)
            rhs_ptr_imag += remaining_cols*offsetB;
          for(Index k = 0; k < depth; k++)
          {
            for(Index arow = 0; arow < remaining_rows; arow++)
            {
              Scalar lhs_real = lhs_ptr[arow];
              Scalar lhs_imag;
              if(!LhsIsReal) lhs_imag = lhs_ptr_imag[arow];
              Scalarc lhsc;

              lhsc.real(lhs_real);
              if(!LhsIsReal)
              {
                if(ConjugateLhs) 
                  lhsc.imag(-lhs_imag);
                else
                  lhsc.imag(lhs_imag);
              } else {
                lhsc.imag((Scalar)0);
              }

              for(Index acol = 0; acol < remaining_cols; acol++ )
              {
                Scalar rhs_real = rhs_ptr[acol];
                Scalar rhs_imag;
                if(!RhsIsReal) rhs_imag = rhs_ptr_imag[acol];
                Scalarc rhsc;

                rhsc.real(rhs_real);
                if(!RhsIsReal)
                {
                  if(ConjugateRhs)
                    rhsc.imag(-rhs_imag);
                  else
                    rhsc.imag(rhs_imag);
                } else {
                  rhsc.imag((Scalar)0);
                }
                res(row + arow, col + acol) += alpha*lhsc*rhsc;
              }
            }
            rhs_ptr += remaining_cols;
            lhs_ptr += remaining_rows;
            if(!LhsIsReal)
              lhs_ptr_imag += remaining_rows;
            if(!RhsIsReal)
              rhs_ptr_imag += remaining_cols;
          }
        }
      }
}

#pragma GCC reset_options
} // end namespace internal

} // end namespace Eigen
#endif // EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H

