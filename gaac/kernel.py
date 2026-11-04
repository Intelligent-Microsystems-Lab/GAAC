"""Fused decode attention over the compressed KV (tensor cores, head_dim 128, GQA group size 1..8).

The compressed storage order is the mma.sync m16n8k16 fragment order, so each lane's reconstructed BF16 K/V values
are its mma operand registers: no shuffles or shared-memory staging of K/V.  K/V are the A operand and Q^T / P^T the
B operand (score tile = 16 tokens x 8 query heads); register footprint is independent of the group size.
CTA = (batch row, kv head, split); 4 warps stream 16-token tiles with a 3-stage prefetch (tags -> escape bytes ->
consume).  Tier selection is branch-free; the rare raw-exponent groups take a __noinline__ slow path.  A second
kernel attends over the raw BF16 tail (< 16 tokens per head), and a combine kernel merges the split partials.

Cache heads are addressed through a slot map: batch row b reads cache head SLOT[b]*n_kv + kvh (SLOT[b] < 0 = padded
row, output 0), so a serving engine can keep each sequence in a fixed slot while the batch composition changes.
Set TORCH_CUDA_ARCH_LIST="12.1" (DGX Spark) if the default target is wrong.
"""
import torch
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
typedef unsigned long long u64; typedef unsigned int u32;

__device__ __forceinline__ void mma16816(float* c,const u32* a,u32 b0,u32 b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
// weighted popcount of tier flag bits below `bits`: tier1 -> 1 byte, tier2 (both bits) -> 5 bytes
__device__ __forceinline__ int ovf_prefix(u32 fl,int bits){ u32 low= bits>=32? fl : (fl&((1u<<bits)-1u)); return __popc(low&0x55555555u)+4*__popc(low&0xAAAAAAAAu); }
// 3-bit code fields of 4 consecutive elements at bit offset p of a 96/192-bit LE word -> 4 bytes
__device__ __forceinline__ u32 codes4(const u32* w,int p){ u32 v=__funnelshift_r(w[p>>5],w[(p>>5)+1],p&31)&0xFFFu;
  return (v&7u)|((v&0x38u)<<5)|((v&0x1C0u)<<10)|((v&0xE00u)<<15); }
// sign/mantissa bytes (4) + exponent bytes (4) -> two bf16x2 words (elems 0,1) and (2,3)
__device__ __forceinline__ void bf16x4(u32 sm4,u32 e4,u32& p01,u32& p23){
  u32 h4=(sm4&0x80808080u)|((e4>>1)&0x7F7F7F7Fu); u32 l4=((e4&0x01010101u)<<7)|(sm4&0x7F7F7F7Fu);
  p01=__byte_perm(l4,h4,0x5140); p23=__byte_perm(l4,h4,0x7362); }
// tier-2 slow path: 5 raw bytes -> 8 exponent high-5-bit fields ; returns exponent bytes for elements [4q,4q+4) of the group
__device__ __noinline__ u32 tier2_e4(const uint8_t* p,u32 code4,int q){
  u64 hi=(u64)p[0]|((u64)p[1]<<8)|((u64)p[2]<<16)|((u64)p[3]<<24)|((u64)p[4]<<32); u32 e=0;
  #pragma unroll
  for(int i=0;i<4;i++) e|=((u32)((hi>>(5*(4*q+i)))&31ull)<<3)<<(8*i);
  return code4|e; }

struct KOct { uint4 sm0,sm1; u32 c0,c1,c2; u32 fl; u32 ext; };        // one lane's share of an 8-token octet (fl = row flags)
struct VTile { uint4 sm[4]; u32 c[6]; u32 fl[4]; u32 ext0,ext1; };    // one lane's share of a 16-token tile
__device__ __forceinline__ int wpopc(u32 fl){ return __popc(fl&0x55555555u)+4*__popc(fl&0xAAAAAAAAu); }

template<int G>
__global__ void __launch_bounds__(128,2) gaac_attn(
  const uint8_t* __restrict__ SMK,const uint8_t* __restrict__ CK,const int* __restrict__ FK,const int* __restrict__ TOK,const uint8_t* __restrict__ OVK,const int* __restrict__ OHSK,const uint8_t* __restrict__ B3K,const uint8_t* __restrict__ B4K,
  const uint8_t* __restrict__ SMV,const uint8_t* __restrict__ CV,const int* __restrict__ FV,const int* __restrict__ TOV,const uint8_t* __restrict__ OVV,const int* __restrict__ OHSV,const uint8_t* __restrict__ B3V,const uint8_t* __restrict__ B4V,
  const __nv_bfloat16* __restrict__ Q, float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PA,
  const int* __restrict__ SLOT,const int* __restrict__ SL,int S_pad,float scale,int NSPLIT,int NSLOT,int n_kv,int n_q,int NB){
  constexpr int D=128;
  const int bh=blockIdx.x, split=blockIdx.y, tid=threadIdx.x, warp=tid>>5, lane=tid&31, j=lane&3, r=lane>>2;
  const int b=bh/n_kv, kvh=bh%n_kv;                 // b: batch row (query / output)
  const int slot=SLOT[b], ch=(slot<0?0:slot)*n_kv+kvh;   // ch: cache head
  const int S= slot<0? 0 : SL[ch];   // compressed tokens of this head (S_pad = capacity stride, multiple of 16)
  const long smh=(long)ch*S_pad*D, cdh=(long)ch*S_pad*(D*3/8), rowb=(long)ch*S_pad;
  const int ohk=OHSK[ch], ohv=OHSV[ch];
  const int ntiles=(S+15)>>4, chunk=(ntiles+NSPLIT-1)/NSPLIT, tlo=split*chunk, thi=min(tlo+chunk,ntiles); const long tileb=(long)ch*(S_pad>>4);
  // ---- Q A-fragments: row g=r (head), k-step s: a0=(g, 16s+2j,+1) a2=(g, 16s+2j+8,+9); rows g+8 are zero
  u32 qa[8][2];
  { const int g=r; const bool ok=g<G; const __nv_bfloat16* qp=Q+((long)(b*n_q+kvh*G+(ok?g:0)))*D;
    #pragma unroll
    for(int s=0;s<8;s++){ qa[s][0]= ok? *reinterpret_cast<const u32*>(qp+16*s+2*j):0u; qa[s][1]= ok? *reinterpret_cast<const u32*>(qp+16*s+2*j+8):0u; } }
  // ---- per-lane base tables in shared memory (byte-packed words): K [j][s], V [r][w]
  __shared__ u32 skb3[4][8], skb4[4][8], svb3[8][4], svb4[8][4];
  const int bsel=ch%NB;   // base-table row: NB=n_kv (calibrated per model) or NB=#cache heads (calibrated per sequence)
  if(tid<32){ int jj=tid&3, ss=tid>>2; int d0=16*ss+2*jj; const uint8_t* p3=B3K+bsel*D; const uint8_t* p4=B4K+bsel*D;   // K lane j owns dims 16s+2j+{0,1,8,9}
    skb3[jj][ss]=(u32)p3[d0]|((u32)p3[d0+1]<<8)|((u32)p3[d0+8]<<16)|((u32)p3[d0+9]<<24);
    skb4[jj][ss]=(u32)p4[d0]|((u32)p4[d0+1]<<8)|((u32)p4[d0+8]<<16)|((u32)p4[d0+9]<<24); }
  else if(tid<64){ int rr=(tid-32)>>2, ww=tid&3; const uint8_t* p3=B3V+bsel*D+rr; const uint8_t* p4=B4V+bsel*D+rr;    // V lane class r owns dims 8n+r
    svb3[rr][ww]=(u32)p3[32*ww]|((u32)p3[32*ww+8]<<8)|((u32)p3[32*ww+16]<<16)|((u32)p3[32*ww+24]<<24);
    svb4[rr][ww]=(u32)p4[32*ww]|((u32)p4[32*ww+8]<<8)|((u32)p4[32*ww+16]<<16)|((u32)p4[32*ww+24]<<24); }
  __syncthreads();

  // swapped formulation: S^T[16 tok x 8 heads] = K[16 tok x 16 d] . Q^T[16 d x 8 heads] ; O^T[16 d x 8 heads] = V^T[16 d x 16 tok] . P^T[16 tok x 8 heads]
  float acc[8][4];   // m-tile mt: c0=O[dim 16mt+r][head 2j] c1=[..][2j+1] c2=O[16mt+r+8][2j] c3=[..][2j+1]
  #pragma unroll
  for(int n=0;n<8;n++){acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.f;}
  float m0=-INFINITY,m1=-INFINITY,l0=0.f,l1=0.f;   // heads 2j, 2j+1 ; l partial over this lane's tokens (r, r+8)

  // ---- loaders -------------------------------------------------------------------------------
  auto load_k_pay=[&](int tile,int o,KOct& K){ long sb=smh+(long)tile*(16*D)+o*(8*D)+32*lane; long cb=cdh+(long)tile*(16*48)+o*(8*48)+12*lane;
    K.sm0=*reinterpret_cast<const uint4*>(SMK+sb); K.sm1=*reinterpret_cast<const uint4*>(SMK+sb+16);
    K.c0=*reinterpret_cast<const u32*>(CK+cb); K.c1=*reinterpret_cast<const u32*>(CK+cb+4); K.c2=*reinterpret_cast<const u32*>(CK+cb+8); };
  auto load_k_meta=[&](int tile,int o)->u32{ int tok=tile*16+o*8+r; return tok<S? (u32)FK[rowb+tok] : 0u; };
  // in-tile overflow prefix for tokens r (octet 0) and r+8 (octet 1): warp scan over r (lanes 4r+j hold identical values)
  auto k_prefix=[&](u32 fl0,u32 fl1,int& p0,int& p1){ int b0=wpopc(fl0), b1=wpopc(fl1); int i0=b0, i1=b1;
    #pragma unroll
    for(int o=4;o<32;o<<=1){ int v0=__shfl_up_sync(0xffffffffu,i0,o), v1=__shfl_up_sync(0xffffffffu,i1,o); if(lane>=o){i0+=v0; i1+=v1;} }
    int tot0=__shfl_sync(0xffffffffu,i0,28+j); p0=i0-b0; p1=tot0+i1-b1; };
  auto load_k_ext=[&](u32 fl,int pos0)->u32{ u32 t4=(fl>>(8*j))&0xFFu; if(!t4) return 0u;   // this lane's 4 groups
    const uint8_t* p=OVK+pos0+ovf_prefix(fl,8*j); u32 e=0u;
    #pragma unroll
    for(int g=0;g<4;g++){ u32 t=(t4>>(2*g))&3u; if(t==1u) e|=((u32)*p)<<(8*g); p+= (t==1u)?1:((t==3u)?5:0); }
    return e; };
  auto load_v_pay=[&](int tile,VTile& V){ long sb=smh+(long)tile*(16*D)+64*lane; long cb=cdh+(long)tile*(16*48)+24*lane;
    #pragma unroll
    for(int t=0;t<4;t++) V.sm[t]=*reinterpret_cast<const uint4*>(SMV+sb+16*t);
    #pragma unroll
    for(int w=0;w<6;w++) V.c[w]=*reinterpret_cast<const u32*>(CV+cb+4*w); };
  auto load_v_meta=[&](int tile,u32* m){ const int toks[4]={2*j,2*j+1,2*j+8,2*j+9};
    #pragma unroll
    for(int t=0;t<4;t++){ int tok=tile*16+toks[t]; m[t]= tok<S? (u32)FV[rowb+tok] : 0u;} };
  // in-tile prefixes for tokens 2j,2j+1,2j+8,2j+9: scan over j within each 4-lane group
  auto v_prefix=[&](const u32* m,int* p){ int b0=wpopc(m[0]),b1=wpopc(m[1]),b2=wpopc(m[2]),b3=wpopc(m[3]); int x=b0+b1, y=b2+b3; int ix=x, iy=y;
    #pragma unroll
    for(int o=1;o<4;o<<=1){ int vx=__shfl_up_sync(0xffffffffu,ix,o), vy=__shfl_up_sync(0xffffffffu,iy,o); if(j>=o){ix+=vx; iy+=vy;} }
    int tot=__shfl_sync(0xffffffffu,ix,(lane&~3)|3); p[0]=ix-x; p[1]=p[0]+b0; p[2]=tot+iy-y; p[3]=p[2]+b2; };
  auto load_v_ext=[&](const u32* m,const int* pre,int base,u32& e0,u32& e1){ e0=e1=0u;   // groups 2r,2r+1 of each token -> byte slot 2t+g
    #pragma unroll
    for(int t=0;t<4;t++){ u32 fl=m[t]; u32 t2=(fl>>(4*r))&0xFu; if(!t2) continue;
      const uint8_t* p=OVV+base+pre[t]+ovf_prefix(fl,4*r);
      #pragma unroll
      for(int g=0;g<2;g++){ u32 tt=(t2>>(2*g))&3u; u32 by=(tt==1u)? (u32)*p : 0u; int slot=2*t+g;
        if(slot<4) e0|=by<<(8*slot); else e1|=by<<(8*(slot-4)); p+=(tt==1u)?1:((tt==3u)?5:0); } } };

  // ---- reconstruction (branch-free tier select; rare raw tier via noinline) ----
  int kpos[2]={0,0}; int vpos[4]={0,0,0,0};   // absolute overflow positions of the CURRENT tile's rows (for the rare raw-tier path)
  auto k_frag=[&](const KOct& K,int o_idx,int s,u32& b0,u32& b1){ const u32 cw[4]={K.c0,K.c1,K.c2,0u};
    u32 code4=codes4(cw,12*s); int g=s>>1; u32 fl=K.fl; u32 t=(fl>>(2*(4*j+g)))&3u;
    u32 eb=(K.ext>>(8*g))&0xFFu; u32 bits=(eb>>(4*(s&1)))&0xFu; u32 x4=((bits&1u)|((bits&2u)<<7)|((bits&4u)<<14)|((bits&8u)<<21))<<3;
    u32 e4=(t==0u? skb3[j][s] : skb4[j][s])+code4+(t==1u? x4 : 0u);
    if(t==3u){ const uint8_t* p=OVK+kpos[o_idx]+ovf_prefix(fl,2*(4*j+g)); e4=tier2_e4(p,code4,s&1); }
    const u32* smw=reinterpret_cast<const u32*>(&K.sm0); u32 sm4= s<4? smw[s] : reinterpret_cast<const u32*>(&K.sm1)[s-4];
    bf16x4(sm4,e4,b0,b1); };
  auto v_words=[&](const VTile& V,int t,int w,u32& u01,u32& u23){ const u32 cw[7]={V.c[0],V.c[1],V.c[2],V.c[3],V.c[4],V.c[5],0u};
    u32 code4=codes4(cw,48*t+12*w); int g=w>>1; u32 fl=V.fl[t]; u32 tt=(fl>>(2*(2*r+g)))&3u;
    int slot=2*t+g; u32 eb= slot<4? (V.ext0>>(8*slot))&0xFFu : (V.ext1>>(8*(slot-4)))&0xFFu; u32 bits=(eb>>(4*(w&1)))&0xFu;
    u32 x4=((bits&1u)|((bits&2u)<<7)|((bits&4u)<<14)|((bits&8u)<<21))<<3;
    u32 e4=(tt==0u? svb3[r][w] : svb4[r][w])+code4+(tt==1u? x4 : 0u);
    if(tt==3u){ const uint8_t* p=OVV+vpos[t]+ovf_prefix(fl,2*(2*r+g)); e4=tier2_e4(p,code4,w&1); }
    u32 sm4=reinterpret_cast<const u32*>(&V.sm[t])[w]; bf16x4(sm4,e4,u01,u23); };

  // ---- pipeline state: tile t (consume), t+4 (payload+ext), t+8 (flags)
  KOct kA[2],kB[2]; VTile vA,vB; u32 kmN[2],vmN[4]; int kposN[2],vposN[4];
  auto positions=[&](int tl,const u32* kf,const u32* vf,int* kp,int* vp){ int p0,p1; k_prefix(kf[0],kf[1],p0,p1); int kb=ohk+TOK[tileb+tl]; kp[0]=kb+p0; kp[1]=kb+p1;
    int pv[4]; v_prefix(vf,pv); int vb=ohv+TOV[tileb+tl];
    #pragma unroll
    for(int t=0;t<4;t++) vp[t]=vb+pv[t]; };
  int tile=tlo+warp;
  if(tile<thi){ load_k_pay(tile,0,kA[0]); load_k_pay(tile,1,kA[1]); load_v_pay(tile,vA);
    kA[0].fl=load_k_meta(tile,0); kA[1].fl=load_k_meta(tile,1); load_v_meta(tile,vA.fl);
    { u32 kf[2]={kA[0].fl,kA[1].fl}; positions(tile,kf,vA.fl,kpos,vpos); }
    kA[0].ext=load_k_ext(kA[0].fl,kpos[0]); kA[1].ext=load_k_ext(kA[1].fl,kpos[1]); { int vb=ohv+TOV[tileb+tile]; int pr[4]; for(int t=0;t<4;t++) pr[t]=vpos[t]-vb; load_v_ext(vA.fl,pr,vb,vA.ext0,vA.ext1); }
    int t1=tile+4; if(t1<thi){ kmN[0]=load_k_meta(t1,0); kmN[1]=load_k_meta(t1,1); load_v_meta(t1,vmN); } }
  const u32 FULL=0xffffffffu;
  for(; tile<thi; tile+=4){
    const int t1=tile+4, t2=tile+8;
    if(t1<thi){ load_k_pay(t1,0,kB[0]); load_k_pay(t1,1,kB[1]); load_v_pay(t1,vB);
      kB[0].fl=kmN[0]; kB[1].fl=kmN[1];
      #pragma unroll
      for(int t=0;t<4;t++) vB.fl[t]=vmN[t];
      positions(t1,kmN,vmN,kposN,vposN);
      kB[0].ext=load_k_ext(kmN[0],kposN[0]); kB[1].ext=load_k_ext(kmN[1],kposN[1]);
      { int vb=ohv+TOV[tileb+t1]; int pr[4]; for(int t=0;t<4;t++) pr[t]=vposN[t]-vb; load_v_ext(vmN,pr,vb,vB.ext0,vB.ext1); }
      if(t2<thi){ kmN[0]=load_k_meta(t2,0); kmN[1]=load_k_meta(t2,1); load_v_meta(t2,vmN); } }
    // ---- scores S^T: one mma per k-step; A = K (tok r from octet 0, tok r+8 from octet 1), B = Q^T
    float sc[4]={0.f,0.f,0.f,0.f};
    #pragma unroll
    for(int s=0;s<8;s++){ u32 a[4]; k_frag(kA[0],0,s,a[0],a[2]); k_frag(kA[1],1,s,a[1],a[3]); mma16816(sc,a,qa[s][0],qa[s][1]); }
    // sc: c0=S[tok r][h 2j] c1=S[r][2j+1] c2=S[r+8][2j] c3=S[r+8][2j+1]
    const int tb=tile*16; const bool v0=tb+r<S, v8=tb+r+8<S;
    float s00= v0? sc[0]*scale:-INFINITY, s01= v0? sc[1]*scale:-INFINITY, s80= v8? sc[2]*scale:-INFINITY, s81= v8? sc[3]*scale:-INFINITY;
    float mt0=fmaxf(s00,s80), mt1=fmaxf(s01,s81);
    #pragma unroll
    for(int o=4;o<32;o<<=1){ mt0=fmaxf(mt0,__shfl_xor_sync(FULL,mt0,o)); mt1=fmaxf(mt1,__shfl_xor_sync(FULL,mt1,o)); }
    float mn0=fmaxf(m0,mt0), mn1=fmaxf(m1,mt1); float mu0=(mn0==-INFINITY)?0.f:mn0, mu1=(mn1==-INFINITY)?0.f:mn1;
    float al0=__expf(m0-mu0), al1=__expf(m1-mu1);
    float p00=__expf(s00-mu0), p80=__expf(s80-mu0), p01=__expf(s01-mu1), p81=__expf(s81-mu1);
    l0=l0*al0+(p00+p80); l1=l1*al1+(p01+p81); m0=mn0; m1=mn1;
    #pragma unroll
    for(int n=0;n<8;n++){acc[n][0]*=al0; acc[n][2]*=al0; acc[n][1]*=al1; acc[n][3]*=al1;}
    // ---- P^T as B operand: b0 = P[tok 2j,2j+1][head r], b1 = P[tok 2j+8,2j+9][head r]
    // this lane publishes W0 = bf16x2(P[r][2j], P[r][2j+1]), W8 = bf16x2(P[r+8][2j], P[r+8][2j+1])
    u32 W0,W8; { __nv_bfloat162 x=__floats2bfloat162_rn(p00,p01), y=__floats2bfloat162_rn(p80,p81); W0=*reinterpret_cast<u32*>(&x); W8=*reinterpret_cast<u32*>(&y); }
    const int srcA=8*j+(r>>1), srcB=srcA+4;      // lane holding tok 2j (A) / tok 2j+1 (B) with heads (r&~1, r|1)
    u32 a0=__shfl_sync(FULL,W0,srcA), a8=__shfl_sync(FULL,W8,srcA), b0w=__shfl_sync(FULL,W0,srcB), b8w=__shfl_sync(FULL,W8,srcB);
    const u32 sel=(r&1)? 0x7632u : 0x5410u;      // pick head r's half: odd -> high halves, even -> low halves
    u32 pb0=__byte_perm(a0,b0w,sel), pb1=__byte_perm(a8,b8w,sel);
    // ---- O^T += V^T . P^T : m-tile mt covers dims 16mt..16mt+15 ; A rows r -> dim 16mt+r (n-tile 2mt), rows r+8 -> 16mt+r+8 (n-tile 2mt+1)
    #pragma unroll
    for(int w=0;w<4;w++){ u32 x01,x23,y01,y23,z01,z23,w01,w23;
      v_words(vA,0,w,x01,x23); v_words(vA,1,w,y01,y23); v_words(vA,2,w,z01,z23); v_words(vA,3,w,w01,w23);
      u32 a[4];
      // n-tiles 4w (elems e0) and 4w+1 (e1) -> m-tile 2w ; 4w+2 (e2), 4w+3 (e3) -> m-tile 2w+1
      a[0]=__byte_perm(x01,y01,0x5410); a[2]=__byte_perm(z01,w01,0x5410);   // dim 8(4w)+r   : (tok 2j,2j+1) , (tok 2j+8,2j+9)
      a[1]=__byte_perm(x01,y01,0x7632); a[3]=__byte_perm(z01,w01,0x7632);   // dim 8(4w+1)+r
      mma16816(acc[2*w],a,pb0,pb1);
      a[0]=__byte_perm(x23,y23,0x5410); a[2]=__byte_perm(z23,w23,0x5410);   // dim 8(4w+2)+r
      a[1]=__byte_perm(x23,y23,0x7632); a[3]=__byte_perm(z23,w23,0x7632);   // dim 8(4w+3)+r
      mma16816(acc[2*w+1],a,pb0,pb1); }
    kA[0]=kB[0]; kA[1]=kB[1]; vA=vB; kpos[0]=kposN[0]; kpos[1]=kposN[1];
    #pragma unroll
    for(int t=0;t<4;t++) vpos[t]=vposN[t];
  }
  // ---- combine the 4 warps: lane holds O[dims 16mt+r, +8][heads 2j, 2j+1]
  __shared__ float sh_m[4][8], sh_l[4][8], sh_acc[4][8][D];
  #pragma unroll
  for(int o=4;o<32;o<<=1){ l0+=__shfl_xor_sync(FULL,l0,o); l1+=__shfl_xor_sync(FULL,l1,o); }
  if(r==0){ sh_m[warp][2*j]=m0; sh_l[warp][2*j]=l0; sh_m[warp][2*j+1]=m1; sh_l[warp][2*j+1]=l1; }
  #pragma unroll
  for(int n=0;n<8;n++){ sh_acc[warp][2*j][16*n+r]=acc[n][0]; sh_acc[warp][2*j+1][16*n+r]=acc[n][1]; sh_acc[warp][2*j][16*n+r+8]=acc[n][2]; sh_acc[warp][2*j+1][16*n+r+8]=acc[n][3]; }
  __syncthreads();
  { const int d=tid;
    for(int g=0;g<G;g++){ float M=-INFINITY;
      #pragma unroll
      for(int w=0;w<4;w++) M=fmaxf(M,sh_m[w][g]);
      float Lc=0.f,A=0.f;
      #pragma unroll
      for(int w=0;w<4;w++){ float f=(sh_m[w][g]==-INFINITY)?0.f:__expf(sh_m[w][g]-M); Lc+=sh_l[w][g]*f; A+=sh_acc[w][g][d]*f; }
      long qh=(long)(b*n_q+kvh*G+g); long pidx=qh*NSLOT+split;
      if(d==0){PM[pidx]=M; PL[pidx]=Lc;}
      PA[pidx*D+d]=A; } }
}

// raw bf16 tail (0..TCAP tokens per head) -> partial (m,l,acc) in slot NSPLIT ; 128 threads, 16 lanes/token
template<int G>
__global__ void gaac_tail(const __nv_bfloat16* __restrict__ KT,const __nv_bfloat16* __restrict__ VT,const int* __restrict__ SLOT,const int* __restrict__ TL,int TCAP,
  const __nv_bfloat16* __restrict__ Q,float* __restrict__ PM,float* __restrict__ PL,float* __restrict__ PA,float scale,int NSLOT,int n_kv,int n_q){
  constexpr int D=128; const int bh=blockIdx.x, tid=threadIdx.x, unit=tid>>4, lane16=tid&15, db=lane16*8;
  const int b=bh/n_kv, kvh=bh%n_kv; const int slot=SLOT[b], ch=(slot<0?0:slot)*n_kv+kvh;
  const int T= slot<0? 0 : TL[ch]; const long base=(long)ch*TCAP*D; const unsigned full=0xffffffffu;
  __shared__ float qsh[G][D]; __shared__ float sh_m[G][8], sh_l[G][8], sh_acc[G][8][D];
  for(int g=0;g<G;++g) qsh[g][tid]=__bfloat162float(Q[((long)(b*n_q+kvh*G+g))*D+tid]);
  __syncthreads();
  float qg[G][8];
  #pragma unroll
  for(int g=0;g<G;++g)
    #pragma unroll
    for(int i=0;i<8;i++) qg[g][i]=qsh[g][db+i];
  float m_i[G],l_i[G],acc[G][8];
  #pragma unroll
  for(int g=0;g<G;++g){m_i[g]=-INFINITY;l_i[g]=0.f;
    #pragma unroll
    for(int i=0;i<8;i++)acc[g][i]=0.f;}
  for(int k=0;k<(T+7)>>3;++k){
    int token=unit+(k<<3); bool valid=token<T; float pd[G];
    #pragma unroll
    for(int g=0;g<G;++g)pd[g]=0.f;
    if(valid){ float kr[8];
      #pragma unroll
      for(int i=0;i<8;i++) kr[i]=__bfloat162float(KT[base+(long)token*D+db+i]);
      #pragma unroll
      for(int i=0;i<8;i++){
        #pragma unroll
        for(int g=0;g<G;++g)pd[g]+=qg[g][i]*kr[i];}}
    #pragma unroll
    for(int g=0;g<G;++g){
      #pragma unroll
      for(int off=8;off>0;off>>=1)pd[g]+=__shfl_down_sync(full,pd[g],off,16);
      pd[g]=__shfl_sync(full,pd[g],0,16);}
    if(valid){float vr[8];
      #pragma unroll
      for(int i=0;i<8;i++) vr[i]=__bfloat162float(VT[base+(long)token*D+db+i]);
      #pragma unroll
      for(int g=0;g<G;++g){float sc=pd[g]*scale,mn=fmaxf(m_i[g],sc),be=__expf(sc-mn),al=__expf(m_i[g]-mn);
        l_i[g]=l_i[g]*al+be;
        #pragma unroll
        for(int i=0;i<8;i++)acc[g][i]=acc[g][i]*al+be*vr[i]; m_i[g]=mn;}}
  }
  #pragma unroll
  for(int g=0;g<G;++g){if(lane16==0){sh_m[g][unit]=m_i[g];sh_l[g][unit]=l_i[g];}
    #pragma unroll
    for(int i=0;i<8;i++)sh_acc[g][unit][db+i]=acc[g][i];}
  __syncthreads();
  if(tid<D){
    #pragma unroll
    for(int g=0;g<G;++g){float M=-INFINITY;
      #pragma unroll
      for(int u=0;u<8;u++)M=fmaxf(M,sh_m[g][u]);
      float Lc=0.f,A=0.f;
      #pragma unroll
      for(int u=0;u<8;u++){float f=(sh_m[g][u]==-INFINITY)?0.f:__expf(sh_m[g][u]-M);Lc+=sh_l[g][u]*f;A+=sh_acc[g][u][tid]*f;}
      long qh=(long)(b*n_q+kvh*G+g), pidx=qh*NSLOT+(NSLOT-1);
      if(tid==0){PM[pidx]=M;PL[pidx]=Lc;} PA[pidx*D+tid]=A;}}
}

template<int D>
__global__ void gaac_combine(const float* __restrict__ PM,const float* __restrict__ PL,const float* __restrict__ PA,
                             __nv_bfloat16* __restrict__ Out,int NSPLIT){
  const int qh=blockIdx.x, tid=threadIdx.x;
  __shared__ float sM,sL;
  if(tid==0){ float M=-INFINITY;
    for(int s=0;s<NSPLIT;s++) M=fmaxf(M,PM[qh*NSPLIT+s]);
    float L=0.f; for(int s=0;s<NSPLIT;s++){float pm=PM[qh*NSPLIT+s]; L+=(pm==-INFINITY)?0.f:PL[qh*NSPLIT+s]*__expf(pm-M);}
    sM=M; sL=L; }
  __syncthreads();
  float A=0.f;
  for(int s=0;s<NSPLIT;s++){float pm=PM[qh*NSPLIT+s]; float f=(pm==-INFINITY)?0.f:__expf(pm-sM); A+=PA[((long)qh*NSPLIT+s)*D+tid]*f;}
  Out[(long)qh*D+tid]=__float2bfloat16(sL>0.f?A/sL:0.f);
}

torch::Tensor gaac_attn_launch(
  torch::Tensor smk,torch::Tensor ck,torch::Tensor fk,torch::Tensor tok,torch::Tensor ovk,torch::Tensor ohsk,torch::Tensor b3k,torch::Tensor b4k,
  torch::Tensor smv,torch::Tensor cv,torch::Tensor fv,torch::Tensor tov,torch::Tensor ovv,torch::Tensor ohsv,torch::Tensor b3v,torch::Tensor b4v,
  torch::Tensor q,torch::Tensor slot,int n_q,int n_kv,torch::Tensor SL,int S_pad,double scale,int nsplit,
  torch::Tensor kt,torch::Tensor vt,torch::Tensor TL,int TCAP){
  const int D=128; const int G=n_q/n_kv; TORCH_CHECK(G>=1&&G<=8&&G*n_kv==n_q,"GQA group size must be 1..8");
  TORCH_CHECK(q.size(-1)==D,"head_dim must be 128");
  const int B=slot.size(0);
  const bool has_tail=kt.numel()>0; const int NSLOT=nsplit+(has_tail?1:0);
  auto fopt=q.options().dtype(torch::kFloat32);
  auto out=torch::empty({(long)B*n_q,D},q.options());
  auto PM=torch::empty({(long)B*n_q*NSLOT},fopt),PL=torch::empty({(long)B*n_q*NSLOT},fopt),PA=torch::empty({(long)B*n_q*NSLOT*D},fopt);
  auto stream=c10::cuda::getCurrentCUDAStream();   // explicit stream: the legacy default stream is skipped by CUDA-graph capture
  dim3 grid(B*n_kv,nsplit), block(128);
  auto Qp=(const __nv_bfloat16*)q.data_ptr();
  #define ARGS smk.data_ptr<uint8_t>(),ck.data_ptr<uint8_t>(),fk.data_ptr<int>(),tok.data_ptr<int>(),ovk.data_ptr<uint8_t>(),ohsk.data_ptr<int>(),b3k.data_ptr<uint8_t>(),b4k.data_ptr<uint8_t>(),\
               smv.data_ptr<uint8_t>(),cv.data_ptr<uint8_t>(),fv.data_ptr<int>(),tov.data_ptr<int>(),ovv.data_ptr<uint8_t>(),ohsv.data_ptr<int>(),b3v.data_ptr<uint8_t>(),b4v.data_ptr<uint8_t>(),\
               Qp,PM.data_ptr<float>(),PL.data_ptr<float>(),PA.data_ptr<float>(),slot.data_ptr<int>(),SL.data_ptr<int>(),S_pad,(float)scale,nsplit,NSLOT,n_kv,n_q,(int)b3k.size(0)
  switch(G){ case 1: gaac_attn<1><<<grid,block,0,stream>>>(ARGS); break; case 2: gaac_attn<2><<<grid,block,0,stream>>>(ARGS); break;
             case 3: gaac_attn<3><<<grid,block,0,stream>>>(ARGS); break; case 4: gaac_attn<4><<<grid,block,0,stream>>>(ARGS); break;
             case 5: gaac_attn<5><<<grid,block,0,stream>>>(ARGS); break; case 6: gaac_attn<6><<<grid,block,0,stream>>>(ARGS); break;
             case 7: gaac_attn<7><<<grid,block,0,stream>>>(ARGS); break; default: gaac_attn<8><<<grid,block,0,stream>>>(ARGS); }
  if(has_tail){
    #define TARGS (const __nv_bfloat16*)kt.data_ptr(),(const __nv_bfloat16*)vt.data_ptr(),slot.data_ptr<int>(),TL.data_ptr<int>(),TCAP,Qp,PM.data_ptr<float>(),PL.data_ptr<float>(),PA.data_ptr<float>(),(float)scale,NSLOT,n_kv,n_q
    switch(G){ case 1: gaac_tail<1><<<B*n_kv,128,0,stream>>>(TARGS); break; case 2: gaac_tail<2><<<B*n_kv,128,0,stream>>>(TARGS); break;
               case 3: gaac_tail<3><<<B*n_kv,128,0,stream>>>(TARGS); break; case 4: gaac_tail<4><<<B*n_kv,128,0,stream>>>(TARGS); break;
               case 5: gaac_tail<5><<<B*n_kv,128,0,stream>>>(TARGS); break; case 6: gaac_tail<6><<<B*n_kv,128,0,stream>>>(TARGS); break;
               case 7: gaac_tail<7><<<B*n_kv,128,0,stream>>>(TARGS); break; default: gaac_tail<8><<<B*n_kv,128,0,stream>>>(TARGS); } }
  gaac_combine<128><<<B*n_q,128,0,stream>>>(PM.data_ptr<float>(),PL.data_ptr<float>(),PA.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),NSLOT);
  return out;
}
'''
_CPP = "torch::Tensor gaac_attn_launch(" + ",".join(["torch::Tensor"] * 18) + ",int,int,torch::Tensor,int,double,int,torch::Tensor,torch::Tensor,torch::Tensor,int);"
_mod = None


def _ext():
    global _mod
    if _mod is None:
        _mod = load_inline(name="gaac_attn_v1", cpp_sources=_CPP, cuda_sources=_CUDA, functions=["gaac_attn_launch"],
                           extra_cuda_cflags=["-O3", "--use_fast_math"])
    return _mod


def choose_nsplit(B: int, n_kv: int, S: int, target_ctas: int = 96, min_tok: int = 512, cap: int = 48) -> int:
    """Splits per (row, kv head): ~2 CTAs per SM keeps enough warps resident at small batch / long context."""
    return min(max(1, target_ctas // (B * n_kv)), max(1, S // min_tok), cap)


def attention(q, kp, vp, scale=None, nsplit=None):
    """Decode attention over two Pages (kinds 'k' / 'v', H = B*n_kv heads of equal length).
    q: [B, n_q, D] bf16 (one query token per sequence) -> [B, n_q, D] bf16."""
    B, n_q, Dq = q.shape
    n_kv = kp.H // B
    dev = q.device
    if nsplit is None: nsplit = choose_nsplit(B, n_kv, kp.S)
    slots = torch.arange(B, dtype=torch.int32, device=dev)
    SL = torch.full((kp.H,), kp.S, dtype=torch.int32, device=dev)
    e = torch.empty(0, dtype=torch.int32, device=dev); eq = torch.empty(0, dtype=q.dtype, device=dev)
    out = _ext().gaac_attn_launch(kp.sm, kp.codes, kp.flags, kp.tile_off, kp.ovf, kp.ovf_head_start, kp.base3, kp.base4,
                                  vp.sm, vp.codes, vp.flags, vp.tile_off, vp.ovf, vp.ovf_head_start, vp.base3, vp.base4,
                                  q.reshape(B * n_q, Dq).contiguous(), slots, n_q, n_kv, SL, kp.sm.shape[1] // 128,
                                  scale if scale is not None else Dq ** -0.5, nsplit, eq, eq, e, 0)
    return out.view(B, n_q, Dq)
