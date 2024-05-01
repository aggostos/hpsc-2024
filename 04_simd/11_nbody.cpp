#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
   /* 
    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    */
    // rx, ry
    __m512 xivec = _mm512_set1_ps(x[i]);
    __m512 yivec = _mm512_set1_ps(y[i]);
    __m512 xvec = _mm512_load_ps(x);
    __m512 yvec = _mm512_load_ps(y);
    __m512 rxvec = _mm512_sub_ps(xivec, xvec);
    __m512 ryvec = _mm512_sub_ps(yivec, yvec);
    // 1/r
    __m512 rxyvec = _mm512_add_ps(_mm512_mul_ps(rxvec, rxvec), _mm512_mul_ps(ryvec, ryvec));
    __m512 invrvec = _mm512_rsqrt14_ps(rxyvec);
    // m*(1/r)*(1/r)*(1/r) 
    __m512 mvec = _mm512_load_ps(m);
    __m512 mulvec = _mm512_mul_ps(mvec, _mm512_mul_ps(invrvec, _mm512_mul_ps(invrvec, invrvec)));
    // fx, fy
    __m512 fxivec = _mm512_setzero_ps();
    __m512 fyivec = _mm512_setzero_ps();
    __m512 fxi_tmp = _mm512_mul_ps(rxvec, mulvec);
    __m512 fyi_tmp = _mm512_mul_ps(ryvec, mulvec);
    // Conditional, rxy != 0 (i != j)
    __m512 zero = _mm512_setzero_ps();
    __mmask16 mask = _mm512_cmp_ps_mask(rxyvec, zero, _MM_CMPINT_NE);
    fxivec = _mm512_mask_blend_ps(mask, fxivec, fxi_tmp);
    fyivec = _mm512_mask_blend_ps(mask, fyivec, fyi_tmp);

    fx[i] -= _mm512_reduce_add_ps(fxivec); 
    fy[i] -= _mm512_reduce_add_ps(fyivec);
    
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
