#!/usr/bin/env python3
"""msda_hvx/msda_impl.c with a tinygrad path: flags bit 16 (0x10000) runs tg_dispatch (the generated launch
sequence) on one HVX thread instead of the hand kernel's threads, timed the same way. Usage: make_impl.py <src> <dst>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
proto = """
int tg_dispatch(int NV, int has_vis, int stop, const float* value, const float* ref, const float* loc, const float* attw,
                const unsigned char* vis, float* out);
typedef struct {
  int NV, has_vis, stop;
  const float *value, *ref, *loc, *attw;
  const unsigned char* vis;
  float* out;
  int rc;
} tg_job_t;
static void tg_thread_main(void* arg) {
  tg_job_t* j = (tg_job_t*)arg;
  int locked = qurt_hvx_lock(QURT_HVX_MODE_128B) == 0;
  j->rc = tg_dispatch(j->NV, j->has_vis, j->stop, j->value, j->ref, j->loc, j->attw, j->vis, j->out);
  if (locked) qurt_hvx_unlock();
  qurt_thread_exit(0);
}

AEEResult msda_rpc_run("""
s = s.replace("\nAEEResult msda_rpc_run(", proto, 1)
block = """  if (flags & 0x10000) {
    int stop = (flags >> 17) & 0x7f;
    tg_job_t tj = {a.NV, has_vis > 0, stop ? stop : -1, a.value, a.ref, a.loc, a.attw, a.vis, out, -2};
    qurt_thread_t tid;
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, stacks[0]);
    qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
    qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
    if (qurt_thread_create(&tid, &attr, tg_thread_main, &tj) == QURT_EOK) {
      int st;
      qurt_thread_join(tid, &st);
    }
    *dsp_us = HAP_perf_get_time_us() - t0;
    return tj.rc;
  }
"""
anchor = "  unsigned long long t0 = HAP_perf_get_time_us();\n"
assert s.count(anchor) == 1, "anchor"
s = s.replace(anchor, anchor + block, 1)
open(dst, "w").write(s)
print("wrote", dst)
