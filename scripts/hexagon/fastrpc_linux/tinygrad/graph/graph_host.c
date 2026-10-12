/* Host library for run_graph_e2e.py: enables unsigned modules on the cDSP, then opens the
 * graph_rpc session. The qaic stub (graph_rpc_stub.c) supplies the marshalling. */
#include "graph_rpc.h"
#include "remote.h"

#define GRAPH_DSP_URI "file:///graph_rpc.so?graph_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp"

int graph_host_open(remote_handle64* h) {
  struct remote_rpc_control_unsigned_module cfg = {CDSP_DOMAIN_ID, 1};
  int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &cfg, sizeof(cfg));
  if (rc) return rc;
  return graph_rpc_open(GRAPH_DSP_URI, h);
}
