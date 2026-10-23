# Formal models

TLA+ specifications for the remote executor's control plane.

## RosDiscoveryLease

`RosDiscoveryLease.tla` models runner selection in the auto-discovering ROS2
bridge (`../ros2_bridge.cpp`: `discover()`, `expire_discovery()`, the announce
timer). It covers one subscriber, two runners, and their workers and bridges
crashing and recovering. Time advances in announce periods, and a lease lasts
`L` periods (`discovery_timeout_ms / announce_period_ms`, 15 s / 5 s = 3).

Each config file sets the constants; the spec is the same.

| Config | `Verify` (`verify_discovery`) | `ReflectHealth` | Result |
|---|---|---|---|
| `RosDiscoveryLease.cfg` | TRUE (default) | FALSE (current `announce()`) | all properties hold |
| `NoVerify.cfg` | FALSE | FALSE (current `announce()`) | `StaleBound`, `DeadSelectionEnds`, `ConvergesToLive` violated |
| `NoVerifyReflect.cfg` | FALSE | TRUE (announce the worker's real state) | all properties hold |

`ReflectHealth=TRUE` is a candidate fix, not a change to the code. It makes an
unhealthy bridge announce `ready:false`, which `discover()` ignores, so the
lease expires.

### Properties

- `SelectionVerified` (action): a runner is selected only after its
  verification passed, when verification is on.
- `StaleBound` (invariant): a selected runner whose worker is down is dropped
  within `L+1` ticks.
- `DeadSelectionEnds` (liveness): a selected runner whose worker stays down is
  eventually dropped.
- `ConvergesToLive` (liveness): if some runner is eventually up for good, the
  subscriber eventually selects a live runner. The premise is eventual
  stability, so a bridge that flaps forever does not count.

### Running

Needs Java and `tla2tools.jar` (from the TLA+ releases page). From this
directory:

```sh
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC -workers auto -config RosDiscoveryLease.cfg RosDiscoveryLease.tla
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC -workers auto -config NoVerify.cfg RosDiscoveryLease.tla
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC -workers auto -config NoVerifyReflect.cfg RosDiscoveryLease.tla
```

Each run takes about a second and explores roughly 500 distinct states.

### What the model does not cover

- The subscriber never crashes; only the announcing bridges and workers do.
- Forwarding is not a separate event. `StaleBound` is a proxy for requests
  sent to a dead runner.
- Announcements are abstracted: no loss, no reordering, no replay of
  transient-local history, and no `ttl_ms` minimum.
- The configured endpoint is assumed to be up.
- Wall-clock time is abstracted to announce periods.
