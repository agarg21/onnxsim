/* Routes one msda call to the tinygrad launch sequence generated for its shape (capture_msda.py): TSA is
 * NV = 2 without visibility, SCA is NV = 6 with it. Each generated run is one TU (its kernels.c included).
 * tg_stop (TG_PROF builds only): run the first tg_stop launches, for per-launch profiling by prefix. */
#include <stddef.h>

#ifdef TG_PROF
int tg_stop = -1;
#endif

void tg_l0_tsa_run(const float* value, const float* ref, const float* off, const float* attw,
                   const unsigned char* vis_u8, float* out);
void tg_l0_sca_run(const float* value, const float* ref, const float* off, const float* attw,
                   const unsigned char* vis_u8, float* out);

int tg_dispatch(int NV, int has_vis, int stop, const float* value, const float* ref, const float* loc,
                const float* attw, const unsigned char* vis, float* out) {
#ifdef TG_PROF
  tg_stop = stop;
#else
  (void)stop;
#endif
  if (!has_vis && NV == 2) { tg_l0_tsa_run(value, ref, loc, attw, 0, out); return 0; }
  if (has_vis && NV == 6) { tg_l0_sca_run(value, ref, loc, attw, vis, out); return 0; }
  return -3;
}
