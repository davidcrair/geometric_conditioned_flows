 Dead functions/classes (no callers anywhere)

  ┌─────────────────────────┬─────────────────────────────────────────┬─────────────────────────────────────┐
  │          Item           │                Location                 │                Notes                │
  ├─────────────────────────┼─────────────────────────────────────────┼─────────────────────────────────────┤
  │ _roundtrip_to_space()   │ analysis/benchmarks/metric_space.py:160 │ Was used before base-space          │
  │                         │                                         │ predictions                         │
  ├─────────────────────────┼─────────────────────────────────────────┼─────────────────────────────────────┤
  │ load_pickle()           │ data/accessors.py:118                   │ Unused utility                      │
  ├─────────────────────────┼─────────────────────────────────────────┼─────────────────────────────────────┤
  │ CondFlow.step()         │ models/flow.py:142                      │ Unused alternative to forward pass  │
  ├─────────────────────────┼─────────────────────────────────────────┼─────────────────────────────────────┤
  │ CondMeanFlowODE class   │ models/mean_flow.py:109                 │ Superseded by sample_one_step       │
  ├─────────────────────────┼─────────────────────────────────────────┼─────────────────────────────────────┤
  │ sample_mean_flow()      │ models/mean_flow.py:140                 │ Malformed standalone function       │
  ├─────────────────────────┼─────────────────────────────────────────┼─────────────────────────────────────┤
  │ EnergyDistanceLoss      │ training/losses.py:383                  │ Never registered or instantiated    │
  │ class                   │                                         │                                     │
  └─────────────────────────┴─────────────────────────────────────────┴─────────────────────────────────────┘

  Dead imports

  ┌────────────────────────────────┬────────────────────────┐
  │             Import             │        Location        │
  ├────────────────────────────────┼────────────────────────┤
  │ torch.autograd.functional as F │ models/flow.py:10      │
  ├────────────────────────────────┼────────────────────────┤
  │ torchdiffeq.odeint             │ models/mean_flow.py:12 │
  └────────────────────────────────┴────────────────────────┘

  Unused parameters (not dead, but vestigial)

  ┌────────────────────────────────────────────────────┬─────────────────────────┬──────────────────────────┐
  │                        Item                        │        Location         │          Notes           │
  ├────────────────────────────────────────────────────┼─────────────────────────┼──────────────────────────┤
  │ _predictions_in_comparison_space params            │ flow_results.py:117-118 │ Immediately del'd —      │
  │ control_library_size, sample_decode                │                         │ callers still pass them  │
  └────────────────────────────────────────────────────┴─────────────────────────┴──────────────────────────┘

  Backwards-compat wrapper (has callers but just delegates)

  ┌─────────────────────────────────┬─────────────────────┬──────────────────────────────────────────────────┐
  │              Item               │      Location       │                      Notes                       │
  ├─────────────────────────────────┼─────────────────────┼──────────────────────────────────────────────────┤
  │ get_or_build_flow_predictions() │ flow_results.py:365 │ Alias for load_flow_predictions, called from     │
  │                                 │                     │ view_flow_results.ipynb                          │
  └─────────────────────────────────┴─────────────────────┴──────────────────────────────────────────────────┘
