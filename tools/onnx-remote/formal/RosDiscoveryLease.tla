---- MODULE RosDiscoveryLease ----
(***************************************************************************)
(* Model of the auto-discovering ROS2 bridge's runner selection in          *)
(* tools/onnx-remote/ros2_bridge.cpp: discover(), expire_discovery() and    *)
(* the announce timer.                                                     *)
(*                                                                         *)
(* One subscriber (the auto-discovering bridge) watches Runners. Each       *)
(* runner's bridge announces on a period; the subscriber selects one       *)
(* runner, holds it while its lease is alive, and falls back to the        *)
(* configured endpoint (modeled as None) when the lease expires.           *)
(*                                                                         *)
(* Time is discrete: one Tick is one announce period. A lease lasts L      *)
(* ticks (discovery_timeout_ms / announce_period_ms). ReflectHealth=FALSE  *)
(* is the current announce(), which always says ready:true. ReflectHealth  *)
(* =TRUE models announcing the worker's real state (a candidate fix).      *)
(***************************************************************************)
EXTENDS Naturals, RosDiscoveryParams   \* L: lease length in ticks, generated from ros2_bridge.cpp

CONSTANTS
  Runners,        \* bridges that can announce a remote worker
  None,           \* model value: no runner selected (configured endpoint)
  Verify,         \* verify_discovery
  ReflectHealth   \* TRUE: announce ready = worker health; FALSE: always ready

VARIABLES
  workerUp,       \* [Runners -> BOOLEAN]  the remote worker is serving
  bridgeUp,       \* [Runners -> BOOLEAN]  the announcing bridge process is alive
  sel,            \* Runners \cup {None}   the selected runner
  age,            \* ticks since the selected runner's lease was refreshed
  stale,          \* consecutive ticks sel is set while its worker is down
  tog             \* flips on every delivered announcement; keeps Announce non-stuttering

vars == <<workerUp, bridgeUp, sel, age, stale, tog>>

MaxStale == L + 2

Min(a, b) == IF a < b THEN a ELSE b

Ready(r) == workerUp[r] \/ ~ReflectHealth

Init ==
  /\ workerUp = [r \in Runners |-> TRUE]
  /\ bridgeUp = [r \in Runners |-> TRUE]
  /\ sel = None
  /\ age = 0
  /\ stale = 0
  /\ tog = FALSE

----------------------------------------------------------------------------
(* Faults *)

WorkerCrash(r) ==
  /\ workerUp[r]
  /\ workerUp' = [workerUp EXCEPT ![r] = FALSE]
  /\ UNCHANGED <<bridgeUp, sel, age, stale, tog>>

WorkerRecover(r) ==
  /\ ~workerUp[r]
  /\ workerUp' = [workerUp EXCEPT ![r] = TRUE]
  /\ stale' = 0                 \* the outage ends; the stale run restarts
  /\ UNCHANGED <<bridgeUp, sel, age, tog>>

BridgeCrash(r) ==
  /\ bridgeUp[r]
  /\ bridgeUp' = [bridgeUp EXCEPT ![r] = FALSE]
  /\ UNCHANGED <<workerUp, sel, age, stale, tog>>

BridgeRecover(r) ==
  /\ ~bridgeUp[r]
  /\ bridgeUp' = [bridgeUp EXCEPT ![r] = TRUE]
  /\ UNCHANGED <<workerUp, sel, age, stale, tog>>

----------------------------------------------------------------------------
(* Timer: expire_discovery() then the announce tick. Ages the lease and    *)
(* drops it once it is older than L ticks.                                 *)

Tick ==
  /\ IF sel = None
     THEN /\ sel' = None
          /\ age' = 0
     ELSE IF age + 1 > L
          THEN /\ sel' = None
               /\ age' = 0
          ELSE /\ sel' = sel
               /\ age' = age + 1
  /\ stale' = IF sel' # None /\ ~workerUp[sel']
                THEN Min(stale + 1, MaxStale)
                ELSE 0
  /\ UNCHANGED <<workerUp, bridgeUp, tog>>

----------------------------------------------------------------------------
(* Delivery of one announcement from runner r to the subscriber:           *)
(* discover(). Ignored when not ready, and while another runner's lease    *)
(* is alive. A selected runner's announcement refreshes the lease only if  *)
(* it passes verification (when verification is on).                       *)

Announce(r) ==
  /\ bridgeUp[r]
  /\ Ready(r)
  /\ \/ /\ sel = r                                  \* refresh the lease
        /\ (~Verify \/ workerUp[r])
        /\ sel' = sel
        /\ age' = 0
     \/ /\ sel = r                                  \* re-verification fails: rejected
        /\ Verify
        /\ ~workerUp[r]
        /\ UNCHANGED <<sel, age>>
     \/ /\ sel = None                               \* select a new runner
        /\ (~Verify \/ workerUp[r])
        /\ sel' = r
        /\ age' = 0
     \/ /\ sel = None                               \* rejected: verification fails
        /\ Verify
        /\ ~workerUp[r]
        /\ UNCHANGED <<sel, age>>
     \/ /\ sel # None                               \* another runner holds a live lease
        /\ sel # r
        /\ UNCHANGED <<sel, age>>
  /\ tog' = ~tog
  /\ UNCHANGED <<workerUp, bridgeUp, stale>>

----------------------------------------------------------------------------

Next ==
  \/ Tick
  \/ \E r \in Runners:
       \/ Announce(r)
       \/ WorkerCrash(r)
       \/ WorkerRecover(r)
       \/ BridgeCrash(r)
       \/ BridgeRecover(r)

\* Tick is always enabled and runs without fairness pressure from others.
\* Each bridge's announcements are delivered eventually while it is alive.
Spec ==
  /\ Init
  /\ [][Next]_vars
  /\ WF_vars(Tick)
  /\ \A r \in Runners: WF_vars(Announce(r))

----------------------------------------------------------------------------
(* Properties *)

TypeOK ==
  /\ workerUp \in [Runners -> BOOLEAN]
  /\ bridgeUp \in [Runners -> BOOLEAN]
  /\ sel \in Runners \cup {None}
  /\ age \in 0..L
  /\ stale \in 0..MaxStale
  /\ tog \in BOOLEAN

\* A runner becomes the selection only if its verification passed (when on).
SelectionVerified ==
  [][ (sel' # None /\ sel' # sel) => (Verify => workerUp[sel']) ]_vars

\* Bounded staleness: a dead selected runner is dropped within L+1 ticks.
StaleBound == stale <= L + 1

\* Liveness: a selected runner whose worker stays down is eventually dropped.
DeadSelectionEnds ==
  \A r \in Runners:
    (sel = r /\ ~workerUp[r]) ~> (sel # r \/ workerUp[r])

\* Liveness: if some runner is eventually up for good (bridge and worker),
\* the subscriber eventually selects a live runner. The premise must be
\* eventual stability: a bridge that flaps forever never gets a fair
\* announcement delivered while it is continuously enabled.
ConvergesToLive ==
  (\E r \in Runners: <>[](bridgeUp[r] /\ workerUp[r]))
    => <>(\E r \in Runners: sel = r /\ workerUp[r])

====
