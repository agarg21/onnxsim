/* DSP-side implementation of ew_rpc.idl. Runs on the cDSP (Hexagon, scalar float).
 * Payloads are little-endian float32 packed in byte sequences; loads and stores go
 * through memcpy so arbitrary byte alignment from the marshaller is safe. */
#include <string.h>
#include "ew_rpc.h"
#include "AEEStdErr.h"

#define EW_OP_RELU 0
#define EW_OP_ADD  1
#define EW_OP_SUB  2
#define EW_OP_MUL  3
#define EW_OP_MAX  4
#define EW_OP_MIN  5
#define EW_OP_ABS  6
#define EW_OP_NEG  7

int ew_rpc_open(const char* uri, remote_handle64* h) {
  (void)uri;
  *h = 1;
  return AEE_SUCCESS;
}

int ew_rpc_close(remote_handle64 h) {
  (void)h;
  return AEE_SUCCESS;
}

int ew_rpc_run_add(remote_handle64 h, int32 a, int32 b, int32* result) {
  (void)h;
  *result = a + b;
  return AEE_SUCCESS;
}

int ew_rpc_elementwise(remote_handle64 h, int32 op, const unsigned char* a, int aLen,
                       const unsigned char* b, int bLen, unsigned char* c, int cLen) {
  (void)h;
  if (op < EW_OP_RELU || op > EW_OP_NEG) return AEE_EBADPARM;
  if (aLen < 0 || aLen % 4 != 0 || cLen != aLen) return AEE_EBADPARM;
  int binary = op == EW_OP_ADD || op == EW_OP_SUB || op == EW_OP_MUL ||
               op == EW_OP_MAX || op == EW_OP_MIN;
  if (binary && bLen != aLen) return AEE_EBADPARM;

  for (int i = 0; i < aLen; i += 4) {
    float x, y = 0.0f, r;
    memcpy(&x, a + i, 4);
    if (binary) memcpy(&y, b + i, 4);
    switch (op) {
      case EW_OP_RELU: r = x > 0.0f ? x : 0.0f; break;
      case EW_OP_ADD:  r = x + y; break;
      case EW_OP_SUB:  r = x - y; break;
      case EW_OP_MUL:  r = x * y; break;
      case EW_OP_MAX:  r = x > y ? x : y; break;
      case EW_OP_MIN:  r = x < y ? x : y; break;
      case EW_OP_ABS:  r = x < 0.0f ? -x : x; break;
      default:         r = -x; break;  /* EW_OP_NEG */
    }
    memcpy(c + i, &r, 4);
  }
  return AEE_SUCCESS;
}

static float ld(const unsigned char* base, int idx) {
  float v;
  memcpy(&v, base + 4 * (size_t)idx, 4);
  return v;
}

static void st(unsigned char* base, int idx, float v) {
  memcpy(base + 4 * (size_t)idx, &v, 4);
}

int ew_rpc_matmul_f32(remote_handle64 h, int32 m, int32 k, int32 n, const unsigned char* a, int aLen,
                      const unsigned char* b, int bLen, unsigned char* c, int cLen) {
  (void)h;
  if (m <= 0 || k <= 0 || n <= 0) return AEE_EBADPARM;
  if (aLen != m * k * 4 || bLen != k * n * 4 || cLen != m * n * 4) return AEE_EBADPARM;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float acc = 0.0f;
      for (int p = 0; p < k; p++) acc += ld(a, i * k + p) * ld(b, p * n + j);
      st(c, i * n + j, acc);
    }
  }
  return AEE_SUCCESS;
}

int ew_rpc_conv2d_f32(remote_handle64 h, const int32* p, int pLen, const unsigned char* x, int xLen,
                      const unsigned char* w, int wLen, const unsigned char* bias, int biasLen,
                      unsigned char* y, int yLen) {
  (void)h;
  if (pLen != 14 * 4) return AEE_EBADPARM;
  int32 q[14];
  memcpy(q, p, sizeof(q));
  int N = q[0], C = q[1], H = q[2], W = q[3], M = q[4], kH = q[5], kW = q[6];
  int sH = q[7], sW = q[8], padT = q[9], padL = q[10], padB = q[11], padR = q[12], hasBias = q[13];
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || M <= 0 || kH <= 0 || kW <= 0 || sH <= 0 || sW <= 0) return AEE_EBADPARM;
  int Hp = H + padT + padB, Wp = W + padL + padR;
  if (Hp < kH || Wp < kW) return AEE_EBADPARM;
  int Ho = (Hp - kH) / sH + 1, Wo = (Wp - kW) / sW + 1;
  if (xLen != N * C * H * W * 4 || wLen != M * C * kH * kW * 4 || yLen != N * M * Ho * Wo * 4) return AEE_EBADPARM;
  if (hasBias ? biasLen != M * 4 : biasLen != 0) return AEE_EBADPARM;

  for (int n = 0; n < N; n++)
    for (int m = 0; m < M; m++)
      for (int oh = 0; oh < Ho; oh++)
        for (int ow = 0; ow < Wo; ow++) {
          float acc = hasBias ? ld(bias, m) : 0.0f;
          for (int c = 0; c < C; c++)
            for (int i = 0; i < kH; i++) {
              int ih = oh * sH + i - padT;
              if (ih < 0 || ih >= H) continue;
              for (int j = 0; j < kW; j++) {
                int iw = ow * sW + j - padL;
                if (iw < 0 || iw >= W) continue;
                acc += ld(x, ((n * C + c) * H + ih) * W + iw) * ld(w, ((m * C + c) * kH + i) * kW + j);
              }
            }
          st(y, ((n * M + m) * Ho + oh) * Wo + ow, acc);
        }
  return AEE_SUCCESS;
}

int ew_rpc_maxpool2d_f32(remote_handle64 h, const int32* p, int pLen, const unsigned char* x, int xLen,
                         unsigned char* y, int yLen) {
  (void)h;
  if (pLen != 12 * 4) return AEE_EBADPARM;
  int32 q[12];
  memcpy(q, p, sizeof(q));
  int N = q[0], C = q[1], H = q[2], W = q[3], kH = q[4], kW = q[5];
  int sH = q[6], sW = q[7], padT = q[8], padL = q[9], padB = q[10], padR = q[11];
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || kH <= 0 || kW <= 0 || sH <= 0 || sW <= 0) return AEE_EBADPARM;
  int Hp = H + padT + padB, Wp = W + padL + padR;
  if (Hp < kH || Wp < kW) return AEE_EBADPARM;
  int Ho = (Hp - kH) / sH + 1, Wo = (Wp - kW) / sW + 1;
  if (xLen != N * C * H * W * 4 || yLen != N * C * Ho * Wo * 4) return AEE_EBADPARM;

  for (int nc = 0; nc < N * C; nc++)
    for (int oh = 0; oh < Ho; oh++)
      for (int ow = 0; ow < Wo; ow++) {
        int have = 0;
        float best = 0.0f;
        for (int i = 0; i < kH; i++) {
          int ih = oh * sH + i - padT;
          if (ih < 0 || ih >= H) continue;
          for (int j = 0; j < kW; j++) {
            int iw = ow * sW + j - padL;
            if (iw < 0 || iw >= W) continue;
            float v = ld(x, (nc * H + ih) * W + iw);
            if (!have || v > best) best = v;
            have = 1;
          }
        }
        st(y, (nc * Ho + oh) * Wo + ow, best);
      }
  return AEE_SUCCESS;
}
