/* Routes one msda call to the generated tinygrad phases for its shape (capture_msda.py): TSA = NV 2 without visibility,
 * SCA = NV 6 with it, decoder = NV 1 without it. Each case's functions are weak, so a skel holds any subset of the cases
 * it was built with. tg_stop (TG_PROF builds only): run the first tg_stop launches, for per-launch profiling. */
#include <stddef.h>
#define WEAK __attribute__((weak))
#define CASE_DECL(tag)                                                                                             \
  int tg_##tag##_nphase(void) WEAK;                                                                                \
  int tg_##tag##_is_range(int) WEAK;                                                                               \
  int tg_##tag##_phase_ext(int) WEAK;                                                                              \
  void tg_##tag##_prep_call(const unsigned char*) WEAK;                                                            \
  void tg_##tag##_call(const float*, const float*, const float*, const float*, const unsigned char*, float*, int, int, int) WEAK;
CASE_DECL(l0_tsa)
CASE_DECL(l0_sca)
CASE_DECL(dec_l0)

#ifdef TG_PROF
int tg_stop = -1;
#endif

/* -> case id (0 TSA, 1 SCA, 2 decoder) or -1 */
int tg_case(int NV, int has_vis) {
  if (!has_vis && NV == 2) return 0;
  if (has_vis && NV == 6) return 1;
  if (!has_vis && NV == 1) return 2;
  return -1;
}
int tg_nphase(int c) {
  if (c == 0 && tg_l0_tsa_nphase) return tg_l0_tsa_nphase();
  if (c == 1 && tg_l0_sca_nphase) return tg_l0_sca_nphase();
  if (c == 2 && tg_dec_l0_nphase) return tg_dec_l0_nphase();
  return -1;
}
/* a range phase's loop extent (the runner splits [0, extent)); -1 for a skel built before extents were generated */
int tg_phase_ext(int c, int k) {
  if (c == 0 && tg_l0_tsa_phase_ext) return tg_l0_tsa_phase_ext(k);
  if (c == 1 && tg_l0_sca_phase_ext) return tg_l0_sca_phase_ext(k);
  if (c == 2 && tg_dec_l0_phase_ext) return tg_dec_l0_phase_ext(k);
  return -1;
}
int tg_is_range(int c, int k) {
  if (c == 0) return tg_l0_tsa_is_range(k);
  if (c == 1) return tg_l0_sca_is_range(k);
  return tg_dec_l0_is_range(k);
}
void tg_prep(int c, const unsigned char* vis) {
  if (c == 0) tg_l0_tsa_prep_call(0);
  else if (c == 1) tg_l0_sca_prep_call(vis);
  else tg_dec_l0_prep_call(0);
}
void tg_call(int c, const float* value, const float* ref, const float* off, const float* attw, const unsigned char* vis,
             float* out, int k, int q0, int q1) {
#ifdef TG_PROF
  tg_stop = -1;
#endif
  if (c == 0) tg_l0_tsa_call(value, ref, off, attw, 0, out, k, q0, q1);
  else if (c == 1) tg_l0_sca_call(value, ref, off, attw, vis, out, k, q0, q1);
  else tg_dec_l0_call(value, ref, off, attw, 0, out, k, q0, q1);
}
