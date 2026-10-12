/* Host-side client for ew_rpc. Drives the cDSP through libcdsprpc only (no TVM, no QNN).
 * Checks every elementwise op bit-for-bit against the same expression evaluated on the
 * host. Exits 0 when every case matches. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ew_rpc.h"
#include "remote.h"

#define DSP_URI "file:///ew_rpc.so?ew_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp"

static const char* kOpNames[] = {"relu", "add", "sub", "mul", "max", "min", "abs", "neg"};
static const int kBinary[] = {0, 1, 1, 1, 1, 1, 0, 0};

static float host_op(int op, float x, float y) {
  switch (op) {
    case 0: return x > 0.0f ? x : 0.0f;
    case 1: return x + y;
    case 2: return x - y;
    case 3: return x * y;
    case 4: return x > y ? x : y;
    case 5: return x < y ? x : y;
    case 6: return x < 0.0f ? -x : x;
    default: return -x;
  }
}

static float frand(void) { return ((float)rand() / (float)RAND_MAX) * 200.0f - 100.0f; }

int main(void) {
  struct remote_rpc_control_unsigned_module cfg = {CDSP_DOMAIN_ID, 1};
  int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &cfg, sizeof(cfg));
  if (rc) { printf("[FAIL] enable unsigned module rc=0x%x\n", rc); return 1; }

  remote_handle64 h = 0;
  rc = ew_rpc_open(DSP_URI, &h);
  if (rc) { printf("[FAIL] ew_rpc_open rc=0x%x\n", rc); return 1; }

  int failures = 0;
  int32 sum = 0;
  rc = ew_rpc_run_add(h, 40, 2, &sum);
  if (rc || sum != 42) { printf("[FAIL] run_add(40,2)=%d rc=0x%x\n", sum, rc); failures++; }
  else printf("[ OK ] run_add(40,2) = %d\n", sum);

  const int sizes[] = {1, 7, 1024, 65537};
  for (int s = 0; s < (int)(sizeof(sizes) / sizeof(sizes[0])); s++) {
    int n = sizes[s];
    int bytes = n * 4;
    float* a = malloc(bytes);
    float* b = malloc(bytes);
    float* c = malloc(bytes);
    float* want = malloc(bytes);
    for (int i = 0; i < n; i++) { a[i] = frand(); b[i] = frand(); }
    for (int op = 0; op < 8; op++) {
      for (int i = 0; i < n; i++) want[i] = host_op(op, a[i], kBinary[op] ? b[i] : 0.0f);
      rc = ew_rpc_elementwise(h, op, (const unsigned char*)a, bytes,
                              kBinary[op] ? (const unsigned char*)b : NULL, kBinary[op] ? bytes : 0,
                              (unsigned char*)c, bytes);
      int ok = rc == 0 && memcmp(c, want, bytes) == 0;
      if (!ok) {
        failures++;
        int first = -1;
        for (int i = 0; i < n && first < 0; i++) if (memcmp(&c[i], &want[i], 4)) first = i;
        printf("[FAIL] %-4s n=%-6d rc=0x%x first_mismatch=%d\n", kOpNames[op], n, rc, first);
      } else {
        printf("[ OK ] %-4s n=%-6d bit-exact\n", kOpNames[op], n);
      }
    }
    free(a); free(b); free(c); free(want);
  }

  /* Error paths must be rejected on the DSP, not silently computed. */
  float one = 1.0f, two = 2.0f, out = 0.0f;
  rc = ew_rpc_elementwise(h, 99, (const unsigned char*)&one, 4, NULL, 0, (unsigned char*)&out, 4);
  if (rc == 0) { printf("[FAIL] op=99 accepted\n"); failures++; } else printf("[ OK ] op=99 rejected rc=0x%x\n", rc);
  rc = ew_rpc_elementwise(h, 1, (const unsigned char*)&one, 4, (const unsigned char*)&two, 8, (unsigned char*)&out, 4);
  if (rc == 0) { printf("[FAIL] mismatched add lengths accepted\n"); failures++; } else printf("[ OK ] mismatched lengths rejected rc=0x%x\n", rc);

  rc = ew_rpc_close(h);
  if (rc) { printf("[FAIL] ew_rpc_close rc=0x%x\n", rc); failures++; }
  printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
