#!/usr/bin/env python3
"""msda_hvx/msda_impl.c with a tinygrad path: flags bit 16 (0x10000) runs the generated tinygrad sequence instead of the
hand kernel. The low flag bits are the thread count, as for the hand kernel: one prep, then the queries split into that many
ranges, each on its own HVX thread. Bits 17-23: a launch prefix (TG_PROF builds, profiling). Usage: make_impl.py <src> <dst>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
proto = """
int tg_run(int mode, int NV, int has_vis, int stop, const float* value, const float* ref, const float* loc, const float* attw,
           const unsigned char* vis, float* out, int q0, int q1);
typedef struct {
  int mode, NV, has_vis, stop, q0, q1, rc;
  const float *value, *ref, *loc, *attw;
  const unsigned char* vis;
  float* out;
} tg_job_t;
static void tg_thread_main(void* arg) {
  tg_job_t* j = (tg_job_t*)arg;
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  j->rc = tg_run(j->mode, j->NV, j->has_vis, j->stop, j->value, j->ref, j->loc, j->attw, j->vis, j->out, j->q0, j->q1);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}
static int tg_spawn(tg_job_t* j, int slot) {
  qurt_thread_t tid;
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_stack_addr(&attr, stacks[slot]);
  qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
  qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
  if (qurt_thread_create(&tid, &attr, tg_thread_main, j) != QURT_EOK) return -2;
  int st;
  qurt_thread_join(tid, &st);
  return j->rc;
}

AEEResult msda_rpc_run("""
assert s.count("\nAEEResult msda_rpc_run(") == 1
s = s.replace("\nAEEResult msda_rpc_run(", proto, 1)
block = """  if (flags & 0x10000) {
    int stop = (flags >> 17) & 0x7f;
    stop = stop ? stop : -1;
    int nt = flags & 0xff;
    if (nt < 1) nt = 1;
    if (nt > MAX_THREADS) nt = MAX_THREADS;
    int Q = a.Q;
    /* prep once (constants, the visibility mask), then the query ranges in parallel */
    tg_job_t pj = {0, a.NV, has_vis > 0, stop, 0, 0, -2, a.value, a.ref, a.loc, a.attw, a.vis, out};
    int rc = tg_spawn(&pj, 0);
    /* concurrent ranges */
    {
      static tg_job_t rjobs[MAX_THREADS];
      qurt_thread_t tids[MAX_THREADS] = {0};
      for (int t = 0; t < nt && rc == 0; t++) {
        rjobs[t] = (tg_job_t){1, a.NV, has_vis > 0, stop, (int)((long)Q * t / nt), (int)((long)Q * (t + 1) / nt), -2,
                              a.value, a.ref, a.loc, a.attw, a.vis, out};
        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
        qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
        qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
        if (qurt_thread_create(&tids[t], &attr, tg_thread_main, &rjobs[t]) != QURT_EOK) { rc = -2; break; }
      }
      for (int t = 0; t < nt; t++) { int st; if (tids[t]) qurt_thread_join(tids[t], &st); }
      for (int t = 0; t < nt && rc == 0; t++) if (rjobs[t].rc) rc = rjobs[t].rc;
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
