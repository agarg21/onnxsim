#!/usr/bin/env python3
"""msda_hvx/msda_impl.c with a tinygrad path: flags bit 16 (0x10000) runs the generated tinygrad phases instead of the hand
kernel. The low flag bits are the thread count, as for the hand kernel: the prep runs once, then each phase in order; a
query-range phase splits its queries across the threads (joined before the next phase), a whole-buffer phase runs on one.
Usage: make_impl.py <src> <dst>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
proto = """
int tg_case(int NV, int has_vis);
int tg_nphase(int c);
int tg_is_range(int c, int k);
void tg_prep(int c, const unsigned char* vis);
void tg_call(int c, const float* value, const float* ref, const float* off, const float* attw, const unsigned char* vis,
             float* out, int k, int q0, int q1);
typedef struct {
  int mode, c, k, q0, q1, rc;   /* mode 0: prep; mode 1: phase k over [q0, q1) */
  const float *value, *ref, *loc, *attw;
  const unsigned char* vis;
  float* out;
} tg_job_t;
static void tg_thread_main(void* arg) {
  tg_job_t* j = (tg_job_t*)arg;
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  if (j->mode == 0) tg_prep(j->c, j->vis);
  else tg_call(j->c, j->value, j->ref, j->loc, j->attw, j->vis, j->out, j->k, j->q0, j->q1);
  j->rc = 0;
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

AEEResult msda_rpc_run("""
assert s.count("\nAEEResult msda_rpc_run(") == 1
s = s.replace("\nAEEResult msda_rpc_run(", proto, 1)
block = """  if (flags & 0x10000) {
    int nt = flags & 0xff;
    if (nt < 1) nt = 1;
    if (nt > MAX_THREADS) nt = MAX_THREADS;
    int c = tg_case(a.NV, has_vis > 0);
    if (c < 0 || tg_nphase(c) < 0) return -3;
    int Q = a.Q;
    static tg_job_t jobs[MAX_THREADS];
    qurt_thread_t tids[MAX_THREADS] = {0};
    /* run one job per thread slot, all started together, then joined */
    #define TG_RUN_JOBS(n) do { \\
      for (int t = 0; t < (n); t++) { \\
        qurt_thread_attr_t attr; \\
        qurt_thread_attr_init(&attr); \\
        qurt_thread_attr_set_stack_addr(&attr, stacks[t]); \\
        qurt_thread_attr_set_stack_size(&attr, STACK_SIZE); \\
        qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id())); \\
        tids[t] = 0; \\
        if (qurt_thread_create(&tids[t], &attr, tg_thread_main, &jobs[t]) != QURT_EOK) { tids[t] = 0; rc = -2; } \\
      } \\
      for (int t = 0; t < (n); t++) { int st; if (tids[t]) qurt_thread_join(tids[t], &st); } \\
    } while (0)
    int rc = 0;
    #define TG_JOB(i, m, kk, a0, a1) jobs[i] = (tg_job_t){m, c, kk, a0, a1, 0, a.value, a.ref, a.loc, a.attw, a.vis, out}
    TG_JOB(0, 0, 0, 0, 0);
    TG_RUN_JOBS(1);
    int np = tg_nphase(c);
    for (int k = 0; k < np && rc == 0; k++) {
      if (tg_is_range(c, k)) {
        for (int t = 0; t < nt; t++) TG_JOB(t, 1, k, (int)((long)Q * t / nt), (int)((long)Q * (t + 1) / nt));
        TG_RUN_JOBS(nt);
      } else {
        TG_JOB(0, 1, k, 0, 0);
        TG_RUN_JOBS(1);
      }
    }
    *dsp_us = HAP_perf_get_time_us() - t0;
    return rc;
  }
"""
anchor = "  unsigned long long t0 = HAP_perf_get_time_us();\n"
assert s.count(anchor) == 1, "anchor"
s = s.replace(anchor, anchor + block, 1)
open(dst, "w").write(s)
print("wrote", dst)
