namespace Eigen {

namespace internal {

template<typename Scalar, typename Packet, typename DataMapper, typename Index, const Index accRows>
EIGEN_STRONG_INLINE void gemm_extra_col(
  const DataMapper& res,
  const Scalar *lhs_base,
  const Scalar *rhs_base,
  Index depth,
  Index strideA,
  Index offsetA,
  Index row,
  Index col,
  Index remaining_rows,
  Index remaining_cols,
  const Packet& pAlpha);

template<typename Scalar, typename Packet, typename DataMapper, typename Index, const Index accRows>
EIGEN_STRONG_INLINE void gemm_extra_row(
  const DataMapper& res,
  const Scalar *lhs_base,
  const Scalar *rhs_base,
  Index depth,
  Index strideA,
  Index offsetA,
  Index row,
  Index col,
  Index cols,
  Index remaining_rows,
  const Packet& pAlpha,
  const Packet& pMask);

template<typename Scalar, typename Packet, typename DataMapper, typename Index, const Index accCols>
EIGEN_STRONG_INLINE void gemm_unrolled_col(
  const DataMapper& res,
  const Scalar *lhs_base,
  const Scalar *rhs_base,
  Index depth,
  Index strideA,
  Index offsetA,
  Index& row,
  Index rows,
  Index col,
  Index remaining_cols,
  const Packet& pAlpha);

template<typename Packet>
EIGEN_STRONG_INLINE Packet bmask(const int remaining_rows);

const static Packet16uc p16uc_SETCOMPLEX32_FIRST = {  0,  1,  2,  3,
                                                     16, 17, 18, 19,
                                                      4,  5,  6,  7,
                                                     20, 21, 22, 23};

const static Packet16uc p16uc_SETCOMPLEX32_SECOND = {  8,  9, 10, 11,
                                                      24, 25, 26, 27,
                                                      12, 13, 14, 15,
                                                      28, 29, 30, 31};
//[a,b],[ai,bi] = [a,ai] - This is equivalent to p16uc_GETREAL64
const static Packet16uc p16uc_SETCOMPLEX64_FIRST = {  0,  1,  2,  3,  4,  5,  6,  7,
                                                     16, 17, 18, 19, 20, 21, 22, 23};

//[a,b],[ai,bi] = [b,bi] - This is equivalent to p16uc_GETIMAG64
const static Packet16uc p16uc_SETCOMPLEX64_SECOND = {  8,  9, 10, 11, 12, 13, 14, 15,
                                                      24, 25, 26, 27, 28, 29, 30, 31};


// Grab two decouples real/imaginary PacketBlocks and return two coupled (real/imaginary pairs) PacketBlocks.
template<typename Packet, typename Packetc>
EIGEN_STRONG_INLINE void bcouple(PacketBlock<Packet,4>& taccReal, PacketBlock<Packet,4>& taccImag, PacketBlock<Packetc,8>& tRes, PacketBlock<Packetc, 4>& acc1, PacketBlock<Packetc, 4>& acc2)
{
  acc1.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], p16uc_SETCOMPLEX32_FIRST);
  acc1.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], p16uc_SETCOMPLEX32_FIRST);
  acc1.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], p16uc_SETCOMPLEX32_FIRST);
  acc1.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], p16uc_SETCOMPLEX32_FIRST);

  acc2.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], p16uc_SETCOMPLEX32_SECOND);
  acc2.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], p16uc_SETCOMPLEX32_SECOND);
  acc2.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], p16uc_SETCOMPLEX32_SECOND);
  acc2.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], p16uc_SETCOMPLEX32_SECOND);

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
EIGEN_STRONG_INLINE void bcouple<Packet2d, Packet1cd>(PacketBlock<Packet2d,4>& taccReal, PacketBlock<Packet2d,4>& taccImag, PacketBlock<Packet1cd,8>& tRes, PacketBlock<Packet1cd, 4>& acc1, PacketBlock<Packet1cd, 4>& acc2)
{
  acc1.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], p16uc_SETCOMPLEX64_FIRST);
  acc1.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], p16uc_SETCOMPLEX64_FIRST);
  acc1.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], p16uc_SETCOMPLEX64_FIRST);
  acc1.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], p16uc_SETCOMPLEX64_FIRST);

  acc2.packet[0].v = vec_perm(taccReal.packet[0], taccImag.packet[0], p16uc_SETCOMPLEX64_SECOND);
  acc2.packet[1].v = vec_perm(taccReal.packet[1], taccImag.packet[1], p16uc_SETCOMPLEX64_SECOND);
  acc2.packet[2].v = vec_perm(taccReal.packet[2], taccImag.packet[2], p16uc_SETCOMPLEX64_SECOND);
  acc2.packet[3].v = vec_perm(taccReal.packet[3], taccImag.packet[3], p16uc_SETCOMPLEX64_SECOND);

  acc1.packet[0] = padd<Packet1cd>(tRes.packet[0], acc1.packet[0]);
  acc1.packet[1] = padd<Packet1cd>(tRes.packet[1], acc1.packet[1]);
  acc1.packet[2] = padd<Packet1cd>(tRes.packet[2], acc1.packet[2]);
  acc1.packet[3] = padd<Packet1cd>(tRes.packet[3], acc1.packet[3]);

  acc2.packet[0] = padd<Packet1cd>(tRes.packet[4], acc2.packet[0]);
  acc2.packet[1] = padd<Packet1cd>(tRes.packet[5], acc2.packet[1]);
  acc2.packet[2] = padd<Packet1cd>(tRes.packet[6], acc2.packet[2]);
  acc2.packet[3] = padd<Packet1cd>(tRes.packet[7], acc2.packet[3]);
}

// This is necessary because ploadRhs for double returns a pair of vectors when MMA is enabled.
template<typename Scalar, typename Packet>
EIGEN_STRONG_INLINE Packet ploadRhs(const Scalar *rhs)
{
    return *((Packet *)rhs);
}

} // end namespace internal
} // end namespace Eigen
