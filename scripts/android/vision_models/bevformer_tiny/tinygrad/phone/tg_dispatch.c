/* Routes one msda call to the tinygrad launch sequence generated for its shape (capture_msda.py): TSA is NV = 2 without
 * visibility, SCA is NV = 6 with it. The generated code has a one-time prep and per-query-range launches; disjoint query
 * ranges are independent, so the skel runs them on several HVX threads. tg_stop (TG_PROF builds only): run the first
 * tg_stop launches, for per-launch profiling by prefix. */
#include <stddef.h>

#ifdef TG_PROF
int tg_stop = -1;
#endif

void tg_l0_tsa_prep(const unsigned char* vis_u8);
void tg_l0_tsa_range(const float* value, const float* ref, const float* off, const float* attw,
                     const unsigned char* vis_u8, float* out, int q0, int q1);
void tg_l0_sca_prep(const unsigned char* vis_u8);
void tg_l0_sca_range(const float* value, const float* ref, const float* off, const float* attw,
                     const unsigned char* vis_u8, float* out, int q0, int q1);

/* mode 0: the one-time prep; mode 1: queries [q0, q1) */
int tg_run(int mode, int NV, int has_vis, int stop, const float* value, const float* ref, const float* loc,
           const float* attw, const unsigned char* vis, float* out, int q0, int q1) {
#ifdef TG_PROF
  tg_stop = stop;
#else
  (void)stop;
#endif
  if (!has_vis && NV == 2) {
    if (mode == 0) tg_l0_tsa_prep(0); else tg_l0_tsa_range(value, ref, loc, attw, 0, out, q0, q1);
    return 0;
  }
  if (has_vis && NV == 6) {
    if (mode == 0) tg_l0_sca_prep(vis); else tg_l0_sca_range(value, ref, loc, attw, vis, out, q0, q1);
    return 0;
  }
  return -3;
}
