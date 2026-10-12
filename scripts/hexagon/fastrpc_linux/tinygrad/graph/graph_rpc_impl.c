/* DSP side of graph_rpc.idl: forwards to tg_graph_run(), generated from tinygrad's kernels. */
#include <string.h>
#include "graph_rpc.h"
#include "AEEStdErr.h"
#include "tg_graph_gen.h"

int graph_rpc_open(const char* uri, remote_handle64* h) {
  (void)uri;
  *h = 1;
  return AEE_SUCCESS;
}

int graph_rpc_close(remote_handle64 h) {
  (void)h;
  return AEE_SUCCESS;
}

int graph_rpc_run_graph(remote_handle64 h, const unsigned char* arena, int aLen, unsigned char* out_arena,
                        int cLen) {
  (void)h;
  if (aLen != TG_ARENA_BYTES || cLen != TG_ARENA_BYTES) return AEE_EBADPARM;
  static unsigned char work[TG_ARENA_BYTES] __attribute__((aligned(128)));
  memcpy(work, arena, TG_ARENA_BYTES);
  tg_graph_run(work);
  memcpy(out_arena, work, TG_ARENA_BYTES);
  return AEE_SUCCESS;
}
