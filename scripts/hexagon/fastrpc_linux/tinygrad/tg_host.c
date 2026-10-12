/* Host library for tg_matmul_test.py. Enables unsigned modules on the cDSP, then opens the
 * tg_rpc session. The qaic stub (tg_rpc_stub.c) supplies the marshalling. */
#include "tg_rpc.h"
#include "remote.h"

#define TG_DSP_URI "file:///tg_rpc.so?tg_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp"

int tg_host_open(remote_handle64* h) {
  struct remote_rpc_control_unsigned_module cfg = {CDSP_DOMAIN_ID, 1};
  int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &cfg, sizeof(cfg));
  if (rc) return rc;
  return tg_rpc_open(TG_DSP_URI, h);
}
