/* Host-side driver for the sparse SCA phases on hexagon-sim: loads the l0_sca case, runs each phase once, and reports the
 * pcycles of each phase (hexagon_sim_read_pcycles). */
#include <stdio.h>
#include <stdlib.h>
/* processor cycles: the C3:2 (PCYCLE) register pair */
static inline unsigned long long read_pcycles(void) {
  unsigned long long c;
  __asm__ volatile("%0 = c3:2" : "=r"(c));
  return c;
}
void tg_l0_sca_prep_call(const unsigned char* vis_u8);
void tg_l0_sca_call(const float* value, const float* ref, const float* off, const float* attw,
                    const unsigned char* vis_u8, float* out, int k, int q0, int q1);
int tg_l0_sca_nphase(void);
int tg_l0_sca_phase_ext(int k);
static void* load(const char* dir, const char* name, size_t bytes) {
  char path[512]; snprintf(path, sizeof path, "%s/%s", dir, name);
  void* p = malloc(bytes); FILE* f = fopen(path, "rb");
  if (!f || fread(p, 1, bytes, f) != bytes) { printf("cannot read %s\n", path); exit(1); }
  fclose(f); return p;
}
int main(int argc, char** argv) {
  const char* dir = "/mnt/data/cache/claude-work/bevphase/cases/l0_sca";
  const int NV = 6, S = 375, E = 256, Q = 2500, M = 8, P = 8, R = 4;
  float* value = load(dir, "value.f32", (size_t)NV * S * E * 4);
  float* ref = load(dir, "ref.f32", (size_t)NV * Q * R * 2 * 4);
  float* loc = load(dir, "loc.f32", (size_t)Q * M * P * 2 * 4);
  float* attw = load(dir, "attw.f32", (size_t)Q * M * P * 4);
  unsigned char* vis = load(dir, "vis.u8", (size_t)NV * Q);
  float* out = calloc((size_t)Q * E, sizeof(float));
  tg_l0_sca_prep_call(vis);
  int np = tg_l0_sca_nphase();
#ifdef NP
  np = NP; /* profiling: run only the first NP phases (built per prefix) */
#endif
  unsigned long long total = 0;
  for (int k = 0; k < np; k++) {
    int ext = tg_l0_sca_phase_ext(k);
    unsigned long long t0 = read_pcycles();
    tg_l0_sca_call(value, ref, loc, attw, vis, out, k, 0, ext);
    unsigned long long t1 = read_pcycles();
    total += t1 - t0;
    printf("phase %d (extent %d): %llu pcycles\n", k, ext, t1 - t0);
  }
  printf("total %llu pcycles, out[0]=%f\n", total, out[0]);
  return 0;
}
