"""Decode attention over vLLM's paged block buffer, head_dim 256 (Qwen3.5/3.6/3.8 hybrids).  A 256-dim head is stored as two
128-dim half-heads in consecutive tile slots (block = 2*n_kv half-heads x {K,V}, each half-head with its own bases).  Warp-pair
design: the two warps of a pair stream the same tiles, each owns one half (its K half -> partial score, its V half -> its 128 output
dims); partial scores meet in shared memory with one 64-thread named barrier per tile, so per-warp registers equal the D=128 kernel."""
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
__global__ void __launch_bounds__(128,2) gaac_vpaged256(
  const uint8_t* __restrict__ SMK,const uint8_t* __restrict__ CK,const int* __restrict__ FK,const uint8_t* __restrict__ OVK,const __nv_bfloat16* __restrict__ RAWK,const uint8_t* __restrict__ B3K,const uint8_t* __restrict__ B4K,
  const uint8_t* __restrict__ SMV,const uint8_t* __restrict__ CV,const int* __restrict__ FV,const uint8_t* __restrict__ OVV,const __nv_bfloat16* __restrict__ RAWV,const uint8_t* __restrict__ B3V,const uint8_t* __restrict__ B4V,
  const int* __restrict__ MT,const int* __restrict__ BT,int bt_stride,int TS,
  const __nv_bfloat16* __restrict__ Q, float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PA,
  const int* __restrict__ SL,float scale,int NSPLIT,int NSLOT,int n_kv,int n_q,int NB,u32* __restrict__ DBG,int mode){
  constexpr int D=128;
  const int bh=blockIdx.x, split=blockIdx.y, tid=threadIdx.x, warp=tid>>5, lane=tid&31, j=lane&3, r=lane>>2;
  const int b=bh/n_kv, kvh=bh%n_kv; const int pair=warp>>1, hf=warp&1; const int hh=bh*2+hf;   // warp pair = one tile stream; hf = the 128-dim half this warp owns
  const int S=SL[b];    // tokens of this sequence (compressed tiles + raw filling tile)
  const int ntiles=(S+15)>>4, chunk=(ntiles+NSPLIT-1)/NSPLIT, tlo=split*chunk, thi=min(tlo+chunk,ntiles);
  const int ohk=0, ohv=0;
  // ---- Q A-fragments: row g=r (head), k-step s: a0=(g, 16s+2j,+1) a2=(g, 16s+2j+8,+9); rows g+8 are zero
  u32 qa[8][2];
  { const int g=r; const bool ok=g<G; const __nv_bfloat16* qp=Q+((long)(b*n_q+kvh*G+(ok?g:0)))*(2*D)+hf*D;
    #pragma unroll
    for(int s=0;s<8;s++){ qa[s][0]= ok? *reinterpret_cast<const u32*>(qp+16*s+2*j):0u; qa[s][1]= ok? *reinterpret_cast<const u32*>(qp+16*s+2*j+8):0u; } }
  // ---- per-lane base tables in shared memory (byte-packed words): K [j][s], V [r][w]
  __shared__ u32 skb3[2][4][8], skb4[2][4][8], svb3[2][8][4], svb4[2][8][4];
  auto bsel_of=[&](int hsel){ return (kvh*2+hsel)%NB; };   // base-table row of half-head hsel (NB = half-heads with their own bases)
  if(tid<64){ const int h2=tid>>5, t=tid&31; const int bs=bsel_of(h2); int jj=t&3, ss=t>>2; int d0=16*ss+2*jj; const uint8_t* p3=B3K+bs*D; const uint8_t* p4=B4K+bs*D;   // K lane j owns dims 16s+2j+{0,1,8,9}
    skb3[h2][jj][ss]=(u32)p3[d0]|((u32)p3[d0+1]<<8)|((u32)p3[d0+8]<<16)|((u32)p3[d0+9]<<24);
    skb4[h2][jj][ss]=(u32)p4[d0]|((u32)p4[d0+1]<<8)|((u32)p4[d0+8]<<16)|((u32)p4[d0+9]<<24); }
  else { const int h2=(tid-64)>>5, t=tid&31; const int bs=bsel_of(h2); int rr=t>>2, ww=t&3; const uint8_t* p3=B3V+bs*D+rr; const uint8_t* p4=B4V+bs*D+rr;    // V lane class r owns dims 8n+r
    svb3[h2][rr][ww]=(u32)p3[32*ww]|((u32)p3[32*ww+8]<<8)|((u32)p3[32*ww+16]<<16)|((u32)p3[32*ww+24]<<24);
    svb4[h2][rr][ww]=(u32)p4[32*ww]|((u32)p4[32*ww+8]<<8)|((u32)p4[32*ww+16]<<16)|((u32)p4[32*ww+24]<<24); }
  __shared__ float xsc[4][3][128]; __shared__ int xseq[4]; if(tid<4) xseq[tid]=0;   // partial-score exchange between the two warps of a pair (double-buffered by tile parity)
  __syncthreads();

  // swapped formulation: S^T[16 tok x 8 heads] = K[16 tok x 16 d] . Q^T[16 d x 8 heads] ; O^T[16 d x 8 heads] = V^T[16 d x 16 tok] . P^T[16 tok x 8 heads]
  float acc[8][4];   // m-tile mt: c0=O[dim 16mt+r][head 2j] c1=[..][2j+1] c2=O[16mt+r+8][2j] c3=[..][2j+1]
  #pragma unroll
  for(int n=0;n<8;n++){acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.f;}
  float m0=-INFINITY,m1=-INFINITY,l0=0.f,l1=0.f;   // heads 2j, 2j+1 ; l partial over this lane's tokens (r, r+8)

  // ---- loaders -------------------------------------------------------------------------------
  auto load_k_pay=[&](int p,int o,KOct& K){ long sb=(long)p*TS+o*(8*D)+32*lane; long cb=(long)p*TS+o*(8*48)+12*lane;
    K.sm0=*reinterpret_cast<const uint4*>(SMK+sb); K.sm1=*reinterpret_cast<const uint4*>(SMK+sb+16);
    K.c0=*reinterpret_cast<const u32*>(CK+cb); K.c1=*reinterpret_cast<const u32*>(CK+cb+4); K.c2=*reinterpret_cast<const u32*>(CK+cb+8); };
  auto load_k_meta=[&](int p,int o)->u32{ return (u32)FK[(long)p*(TS>>2)+o*8+r]; };
  // in-tile overflow prefix for tokens r (octet 0) and r+8 (octet 1): warp scan over r (lanes 4r+j hold identical values)
  auto k_prefix=[&](u32 fl0,u32 fl1,int& p0,int& p1){ int b0=wpopc(fl0), b1=wpopc(fl1); int i0=b0, i1=b1;
    #pragma unroll
    for(int o=4;o<32;o<<=1){ int v0=__shfl_up_sync(0xffffffffu,i0,o), v1=__shfl_up_sync(0xffffffffu,i1,o); if(lane>=o){i0+=v0; i1+=v1;} }
    int tot0=__shfl_sync(0xffffffffu,i0,28+j); p0=i0-b0; p1=tot0+i1-b1; };
  auto load_k_ext=[&](u32 fl,int pos0)->u32{ if(mode==1) return 0u; u32 t4=(fl>>(8*j))&0xFFu; if(!t4) return 0u;   // this lane's 4 groups
    const uint8_t* p=OVK+pos0+ovf_prefix(fl,8*j); u32 e=0u;
    #pragma unroll
    for(int g=0;g<4;g++){ u32 t=(t4>>(2*g))&3u; if(t==1u) e|=((u32)*p)<<(8*g); p+= (t==1u)?1:((t==3u)?5:0); }
    return e; };
  auto load_v_pay=[&](int p,VTile& V){ long sb=(long)p*TS+64*lane; long cb=(long)p*TS+24*lane;
    #pragma unroll
    for(int t=0;t<4;t++) V.sm[t]=*reinterpret_cast<const uint4*>(SMV+sb+16*t);
    #pragma unroll
    for(int w=0;w<6;w++) V.c[w]=*reinterpret_cast<const u32*>(CV+cb+4*w); };
  auto load_v_meta=[&](int p,u32* m){ const int toks[4]={2*j,2*j+1,2*j+8,2*j+9};
    #pragma unroll
    for(int t=0;t<4;t++){ m[t]=(u32)FV[(long)p*(TS>>2)+toks[t]];} };
  // in-tile prefixes for tokens 2j,2j+1,2j+8,2j+9: scan over j within each 4-lane group
  auto v_prefix=[&](const u32* m,int* p){ int b0=wpopc(m[0]),b1=wpopc(m[1]),b2=wpopc(m[2]),b3=wpopc(m[3]); int x=b0+b1, y=b2+b3; int ix=x, iy=y;
    #pragma unroll
    for(int o=1;o<4;o<<=1){ int vx=__shfl_up_sync(0xffffffffu,ix,o), vy=__shfl_up_sync(0xffffffffu,iy,o); if(j>=o){ix+=vx; iy+=vy;} }
    int tot=__shfl_sync(0xffffffffu,ix,(lane&~3)|3); p[0]=ix-x; p[1]=p[0]+b0; p[2]=tot+iy-y; p[3]=p[2]+b2; };
  auto load_v_ext=[&](const u32* m,const int* pre,int base,u32& e0,u32& e1){ e0=e1=0u; if(mode==1) return;   // groups 2r,2r+1 of each token -> byte slot 2t+g
    #pragma unroll
    for(int t=0;t<4;t++){ u32 fl=m[t]; u32 t2=(fl>>(4*r))&0xFu; if(!t2) continue;
      const uint8_t* p=OVV+base+pre[t]+ovf_prefix(fl,4*r);
      #pragma unroll
      for(int g=0;g<2;g++){ u32 tt=(t2>>(2*g))&3u; u32 by=(tt==1u)? (u32)*p : 0u; int slot=2*t+g;
        if(slot<4) e0|=by<<(8*slot); else e1|=by<<(8*(slot-4)); p+=(tt==1u)?1:((tt==3u)?5:0); } } };

  // ---- reconstruction (branch-free tier select; rare raw tier via noinline) ----
  int kpos[2]={0,0}; int vpos[4]={0,0,0,0};   // absolute overflow positions of the CURRENT tile's rows (for the rare raw-tier path)
  auto k_frag=[&](const KOct& K,int o_idx,int s,u32& b0,u32& b1){ const u32 cw[4]={K.c0,K.c1,K.c2,0u};
    u32 code4=codes4(cw,12*s); int g=s>>1; u32 fl= mode==1? 0u : K.fl; u32 t=(fl>>(2*(4*j+g)))&3u;
    u32 eb=(K.ext>>(8*g))&0xFFu; u32 bits=(eb>>(4*(s&1)))&0xFu; u32 x4=((bits&1u)|((bits&2u)<<7)|((bits&4u)<<14)|((bits&8u)<<21))<<3;
    u32 e4=(t==0u? skb3[hf][j][s] : skb4[hf][j][s])+code4+(t==1u? x4 : 0u);
    if(t==3u){ const uint8_t* p=OVK+kpos[o_idx]+ovf_prefix(fl,2*(4*j+g)); e4=tier2_e4(p,code4,s&1); }
    const u32* smw=reinterpret_cast<const u32*>(&K.sm0); u32 sm4= s<4? smw[s] : reinterpret_cast<const u32*>(&K.sm1)[s-4];
    bf16x4(sm4,e4,b0,b1); };
  auto v_words=[&](const VTile& V,int t,int w,u32& u01,u32& u23){ const u32 cw[7]={V.c[0],V.c[1],V.c[2],V.c[3],V.c[4],V.c[5],0u};
    u32 code4=codes4(cw,48*t+12*w); int g=w>>1; u32 fl= mode==1? 0u : V.fl[t]; u32 tt=(fl>>(2*(2*r+g)))&3u;
    int slot=2*t+g; u32 eb= slot<4? (V.ext0>>(8*slot))&0xFFu : (V.ext1>>(8*(slot-4)))&0xFFu; u32 bits=(eb>>(4*(w&1)))&0xFu;
    u32 x4=((bits&1u)|((bits&2u)<<7)|((bits&4u)<<14)|((bits&8u)<<21))<<3;
    u32 e4=(tt==0u? svb3[hf][r][w] : svb4[hf][r][w])+code4+(tt==1u? x4 : 0u);
    if(tt==3u){ const uint8_t* p=OVV+vpos[t]+ovf_prefix(fl,2*(2*r+g)); e4=tier2_e4(p,code4,w&1); }
    u32 sm4=reinterpret_cast<const u32*>(&V.sm[t])[w]; bf16x4(sm4,e4,u01,u23); };

  // ---- pipeline state: tile t (consume), t+4 (payload+ext), t+8 (flags)
  KOct kA[2],kB[2]; VTile vA,vB; u32 kmN[2],vmN[4]; int kposN[2],vposN[4];
  auto positions=[&](int pk_,int pv_,const u32* kf,const u32* vf,int* kp,int* vp){ int p0,p1; k_prefix(kf[0],kf[1],p0,p1); int kb=pk_*TS; kp[0]=kb+p0; kp[1]=kb+p1;
    int pv[4]; v_prefix(vf,pv); int vb=pv_*TS;
    #pragma unroll
    for(int t=0;t<4;t++) vp[t]=vb+pv[t]; };
  // logical tile -> physical tiles; negative entry = raw bf16 tile (~p)
  auto mt_at=[&](int tl,int& ek,int& ev){ int pc=BT[(long)b*bt_stride+tl]*(4*n_kv)+2*(kvh*2+hf); ek=MT[pc]; ev=MT[pc+1]; };   // block = 2*n_kv half-heads x {K,V}
  auto resolve=[&](int tl,int& pk_,int& pv_,bool& rawk,bool& rawv){ int ek,ev; mt_at(tl,ek,ev); rawk=ek<0; rawv=ev<0; pk_= rawk? ~ek: ek; pv_= rawv? ~ev: ev; };
  int pkA,pvA,pkB,pvB; bool rkA,rvA,rkB,rvB; int ekN=0,evN=0;   // ekN/evN: raw table entries of the NEXT-next tile (prefetched)
  auto load_k_all=[&](int tl,KOct* K2,int& pk_,int& pv_,bool& rk_,bool& rv_,int* kp,bool pre,int ek,int ev){
    if(pre){ rk_=ek<0; rv_=ev<0; pk_= rk_? ~ek: ek; pv_= rv_? ~ev: ev; } else resolve(tl,pk_,pv_,rk_,rv_);
    if(!rk_){ load_k_pay(pk_,0,K2[0]); load_k_pay(pk_,1,K2[1]); K2[0].fl=load_k_meta(pk_,0); K2[1].fl=load_k_meta(pk_,1); } else { K2[0].fl=K2[1].fl=0u; }
    { int p0,p1; k_prefix(K2[0].fl,K2[1].fl,p0,p1); int kb=pk_*TS; kp[0]=kb+p0; kp[1]=kb+p1; }
    if(!rk_){ K2[0].ext=load_k_ext(K2[0].fl,kp[0]); K2[1].ext=load_k_ext(K2[1].fl,kp[1]); } else { K2[0].ext=K2[1].ext=0u; } };
  auto load_v_all=[&](int pv_,bool rv_,VTile& V,int* vp){
    if(!rv_){ load_v_pay(pv_,V); load_v_meta(pv_,V.fl); } else { V.fl[0]=V.fl[1]=V.fl[2]=V.fl[3]=0u; }
    { int pv[4]; v_prefix(V.fl,pv); int vb=pv_*TS;
      #pragma unroll
      for(int t=0;t<4;t++) vp[t]=vb+pv[t]; }
    if(!rv_){ int vb=pv_*TS; int pr[4]; for(int t=0;t<4;t++) pr[t]=vp[t]-vb; load_v_ext(V.fl,pr,vb,V.ext0,V.ext1); } else { V.ext0=V.ext1=0u; } };
  // raw tile fragments straight from bf16: K a-frag for token (o*8+r), dims 16s+2j+{0,1} / +8,+9 ; V words for slot t, word w (dims 8(4w+i)+r)
  auto raw_k_frag=[&](int p,int o,int s,u32& b0,u32& b1){ const __nv_bfloat16* kr=RAWK+((long)p*16+o*8+r)*D+16*s+2*j; b0=*reinterpret_cast<const u32*>(kr); b1=*reinterpret_cast<const u32*>(kr+8); };
  auto raw_v_words=[&](int p,int t,int w,u32& u01,u32& u23){ const int toks[4]={2*j,2*j+1,2*j+8,2*j+9}; const __nv_bfloat16* vr=RAWV+((long)p*16+toks[t])*D+8*(4*w)+r;
    unsigned short e0=__bfloat16_as_ushort(vr[0]), e1=__bfloat16_as_ushort(vr[8]), e2=__bfloat16_as_ushort(vr[16]), e3=__bfloat16_as_ushort(vr[24]);
    u01=(u32)e0|((u32)e1<<16); u23=(u32)e2|((u32)e3<<16); };
  int tile=tlo+pair; int it=0; float scp[4]={0.f,0.f,0.f,0.f}; int tbp=0; int pvP=0; bool rvP=false;
  volatile int* xq=xseq;                                       // per-warp publish counters (tiles published so far)
  if(tile<thi){ load_k_all(tile,kA,pkA,pvA,rkA,rvA,kpos,false,0,0); if(tile+2<thi) mt_at(tile+2,ekN,evN); }
  const u32 FULL=0xffffffffu;
  // consume(s4, tb): full scores of a tile -> online softmax + P^T V^T over this warp's V half (vA = V of that tile, pvP/rvP/vpos its raw-tile state)
  auto consume=[&](const float* s4,int tb){
    // sc: full 256-dim scores: c0=S[tok r][h 2j] c1=S[r][2j+1] c2=S[r+8][2j] c3=S[r+8][2j+1]
    const bool v0=tb+r<S, v8=tb+r+8<S;
    float s00= v0? s4[0]*scale:-INFINITY, s01= v0? s4[1]*scale:-INFINITY, s80= v8? s4[2]*scale:-INFINITY, s81= v8? s4[3]*scale:-INFINITY;
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
    if(rvP){
    #pragma unroll
    for(int w=0;w<4;w++){ u32 x01,x23,y01,y23,z01,z23,w01,w23;
      raw_v_words(pvP,0,w,x01,x23); raw_v_words(pvP,1,w,y01,y23); raw_v_words(pvP,2,w,z01,z23); raw_v_words(pvP,3,w,w01,w23);
      u32 a[4];
      // n-tiles 4w (elems e0) and 4w+1 (e1) -> m-tile 2w ; 4w+2 (e2), 4w+3 (e3) -> m-tile 2w+1
      a[0]=__byte_perm(x01,y01,0x5410); a[2]=__byte_perm(z01,w01,0x5410);   // dim 8(4w)+r   : (tok 2j,2j+1) , (tok 2j+8,2j+9)
      a[1]=__byte_perm(x01,y01,0x7632); a[3]=__byte_perm(z01,w01,0x7632);   // dim 8(4w+1)+r
      mma16816(acc[2*w],a,pb0,pb1);
      a[0]=__byte_perm(x23,y23,0x5410); a[2]=__byte_perm(z23,w23,0x5410);   // dim 8(4w+2)+r
      a[1]=__byte_perm(x23,y23,0x7632); a[3]=__byte_perm(z23,w23,0x7632);   // dim 8(4w+3)+r
      mma16816(acc[2*w+1],a,pb0,pb1); }
    } else {
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
    }
  };
  for(; tile<thi; tile+=2, it++){
    const int t1=tile+2, t2=tile+4;
    if(t1<thi){ load_k_all(t1,kB,pkB,pvB,rkB,rvB,kposN,true,ekN,evN); if(t2<thi) mt_at(t2,ekN,evN); }
    load_v_all(pvA,rvA,vB,vposN);                     // V of THIS tile: consumed next iteration (one-tile lag)
    // ---- scores S^T: one mma per k-step; A = K (tok r from octet 0, tok r+8 from octet 1), B = Q^T
    float sc[4]={0.f,0.f,0.f,0.f};
    if(rkA){
      #pragma unroll
      for(int s=0;s<8;s++){ u32 a[4]; raw_k_frag(pkA,0,s,a[0],a[2]); raw_k_frag(pkA,1,s,a[1],a[3]); mma16816(sc,a,qa[s][0],qa[s][1]); }
    } else {
      #pragma unroll
      for(int s=0;s<8;s++){ u32 a[4]; k_frag(kA[0],0,s,a[0],a[2]); k_frag(kA[1],1,s,a[1],a[3]); mma16816(sc,a,qa[s][0],qa[s][1]); } }
    { float* xs=xsc[warp][it%3]; xs[4*lane]=sc[0]; xs[4*lane+1]=sc[1]; xs[4*lane+2]=sc[2]; xs[4*lane+3]=sc[3];
      __syncwarp(); if(lane==0){ __threadfence_block(); xq[warp]=it+1; } }              // publish partial(t); no wait
    if(it>0){ while(xq[warp^1]<it){} __threadfence_block();                              // partner has published partial(t-1)
      const float* xo=xsc[warp^1][(it-1)%3]; scp[0]+=xo[4*lane]; scp[1]+=xo[4*lane+1]; scp[2]+=xo[4*lane+2]; scp[3]+=xo[4*lane+3];
      consume(scp,tbp); }
    scp[0]=sc[0]; scp[1]=sc[1]; scp[2]=sc[2]; scp[3]=sc[3]; tbp=tile*16;
    kA[0]=kB[0]; kA[1]=kB[1]; vA=vB; kpos[0]=kposN[0]; kpos[1]=kposN[1]; pkA=pkB; rkA=rkB; pvP=pvA; rvP=rvA; pvA=pvB; rvA=rvB;
    #pragma unroll
    for(int t=0;t<4;t++) vpos[t]=vposN[t];
  }
  if(it>0){ while(xq[warp^1]<it){} __threadfence_block();                                // drain: the last tile's full score
    const float* xo=xsc[warp^1][(it-1)%3]; scp[0]+=xo[4*lane]; scp[1]+=xo[4*lane+1]; scp[2]+=xo[4*lane+2]; scp[3]+=xo[4*lane+3];
    consume(scp,tbp); }
  // ---- combine the 4 warps: lane holds O[dims 16mt+r, +8][heads 2j, 2j+1]
  __shared__ float sh_m[4][8], sh_l[4][8], sh_acc[4][8][D];
  #pragma unroll
  for(int o=4;o<32;o<<=1){ l0+=__shfl_xor_sync(FULL,l0,o); l1+=__shfl_xor_sync(FULL,l1,o); }
  if(r==0){ sh_m[warp][2*j]=m0; sh_l[warp][2*j]=l0; sh_m[warp][2*j+1]=m1; sh_l[warp][2*j+1]=l1; }
  #pragma unroll
  for(int n=0;n<8;n++){ sh_acc[warp][2*j][16*n+r]=acc[n][0]; sh_acc[warp][2*j+1][16*n+r]=acc[n][1]; sh_acc[warp][2*j][16*n+r+8]=acc[n][2]; sh_acc[warp][2*j+1][16*n+r+8]=acc[n][3]; }
  __syncthreads();
  { for(int g=0;g<G;g++){ float M=-INFINITY;      // warps 2p, 2p+1 (one pair) carry identical (m, l); A0 = dims 0..127 (hf 0), A1 = dims 128..255 (hf 1)
      #pragma unroll
      for(int w=0;w<4;w++) M=fmaxf(M,sh_m[w][g]);
      float Lc=0.f,A0=0.f,A1=0.f;
      #pragma unroll
      for(int p=0;p<2;p++){ float f0=(sh_m[2*p][g]==-INFINITY)?0.f:__expf(sh_m[2*p][g]-M), f1=(sh_m[2*p+1][g]==-INFINITY)?0.f:__expf(sh_m[2*p+1][g]-M);
        Lc+=sh_l[2*p][g]*f0; A0+=sh_acc[2*p][g][tid]*f0; A1+=sh_acc[2*p+1][g][tid]*f1; }
      long qh=(long)(b*n_q+kvh*G+g); long pidx=qh*NSLOT+split;
      if(tid==0){PM[pidx]=M; PL[pidx]=Lc;}
      PA[pidx*(2*D)+tid]=A0; PA[pidx*(2*D)+D+tid]=A1; } }
}

// raw bf16 tail (0..TCAP tokens per head) -> partial (m,l,acc) in slot NSPLIT ; 128 threads, 16 lanes/token like the CUDA-core kernel
template<int D>
__global__ void gaac_combine2(const float* __restrict__ PM,const float* __restrict__ PL,const float* __restrict__ PA,
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

void gaac_vpaged256_launch(
  torch::Tensor buf,torch::Tensor raw,torch::Tensor mt,torch::Tensor bt,torch::Tensor seq_lens,torch::Tensor q,torch::Tensor out,
  torch::Tensor b3k,torch::Tensor b4k,torch::Tensor b3v,torch::Tensor b4v,int TS,int B,int n_q,int n_kv,double scale,int nsplit){
  const int D=256; const int G=n_q/n_kv; TORCH_CHECK(G>=1&&G<=8,"G must be 1..8"); TORCH_CHECK(TS%16==0,"tile stride must be a multiple of 16");
  const int NSLOT=nsplit;
  auto fopt=q.options().dtype(torch::kFloat32);
  auto PM=torch::empty({(long)B*n_q*NSLOT},fopt),PL=torch::empty({(long)B*n_q*NSLOT},fopt),PA=torch::empty({(long)B*n_q*NSLOT*D},fopt);
  dim3 grid(B*n_kv,nsplit), block(128);
  const uint8_t* BUF=buf.data_ptr<uint8_t>(); const __nv_bfloat16* RAW=(const __nv_bfloat16*)raw.data_ptr();
  auto Qp=(const __nv_bfloat16*)q.data_ptr();
  #define ARGS BUF,BUF+2048,(const int*)(BUF+2816),BUF+2880,RAW,b3k.data_ptr<uint8_t>(),b4k.data_ptr<uint8_t>(),\
               BUF,BUF+2048,(const int*)(BUF+2816),BUF+2880,RAW,b3v.data_ptr<uint8_t>(),b4v.data_ptr<uint8_t>(),\
               mt.data_ptr<int>(),bt.data_ptr<int>(),(int)bt.stride(0),TS,Qp,PM.data_ptr<float>(),PL.data_ptr<float>(),PA.data_ptr<float>(),seq_lens.data_ptr<int>(),(float)scale,nsplit,NSLOT,n_kv,n_q,(int)b3k.size(0),nullptr,0
  auto st=c10::cuda::getCurrentCUDAStream();
  switch(G){ case 6: gaac_vpaged256<6><<<grid,block,0,st>>>(ARGS); break; case 8: gaac_vpaged256<8><<<grid,block,0,st>>>(ARGS); break; case 4: gaac_vpaged256<4><<<grid,block,0,st>>>(ARGS); break;
             case 2: gaac_vpaged256<2><<<grid,block,0,st>>>(ARGS); break; case 1: gaac_vpaged256<1><<<grid,block,0,st>>>(ARGS); break; default: TORCH_CHECK(false,"unsupported G"); }
  gaac_combine2<256><<<B*n_q,256,0,st>>>(PM.data_ptr<float>(),PL.data_ptr<float>(),PA.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),NSLOT);
}
'''

_CPP = "void gaac_vpaged256_launch(" + ",".join(["torch::Tensor"] * 11) + ",int,int,int,int,double,int);"
_mod = None


def _ext():
    global _mod
    if _mod is None:
        _mod = load_inline(name="gaac_vpaged256_v1", cpp_sources=_CPP, cuda_sources=_CUDA, functions=["gaac_vpaged256_launch"],
                           extra_cuda_cflags=["-O3", "--use_fast_math"])
    return _mod


def paged_attention256(q, out, buf, raw, mt, bt, seq_lens, bases, TS, n_q, n_kv, nsplit, scale=None):
    """q: [B, n_q, 256]; out: [B, n_q*256]; bases per half-head [2*n_kv, 128]."""
    assert n_q // n_kv <= 8, "GQA group size must be <= 8"
    B = q.shape[0]
    _ext().gaac_vpaged256_launch(buf, raw, mt, bt, seq_lens, q, out, bases[0], bases[1], bases[2], bases[3], TS, B, n_q, n_kv,
                                 float(scale) if scale is not None else 1.0 / (256 ** 0.5), nsplit)
    return out
