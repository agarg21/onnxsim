/* Host shared library for the Python driver (onnx_dsp_run.py). Opens the ew_rpc session on
 * the cDSP after enabling unsigned module loading. The qaic-generated stub (ew_rpc_stub.c)
 * supplies ew_rpc_* marshalling; this file only adds the session setup. */
#include "ew_rpc.h"
#include "remote.h"

#define EW_DSP_URI "file:///ew_rpc.so?ew_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp"

int ew_host_open(remote_handle64* h) {
  struct remote_rpc_control_unsigned_module cfg = {CDSP_DOMAIN_ID, 1};
  int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &cfg, sizeof(cfg));
  if (rc) return rc;
  return ew_rpc_open(EW_DSP_URI, h);
}
