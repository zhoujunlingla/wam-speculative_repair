# Code Review: V7 Action Local Repair

## Scope

Reviewed the V7 diff after adding optional action local repair before teacher fallback.

Changed files reviewed:

- design.md
- evaluation/robotwin/specverify_client_policy.py
- wan_va/wan_va_server.py
- evaluation/robotwin/eval_polict_client_openpi.py
- scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py
- tests/test_specverify_client_policy.py

## Findings

No blocking correctness issues found in the current diff.

## Risks / Notes

- Experiment-validity risk: draft_repair_accept converts chunks that would previously fall back to teacher into repaired draft actions. This should reduce teacher usage and may improve speed, but task success can drop if repair accepts action chunks that full teacher would have solved. The low10 gate must compare success, teacher source rate, and latency against V6 worldlatent.
- Repair payload arrays are stripped from JSONL logs via verify_log; logs retain scalar repair metrics such as repair_delta_mean/max from the server response.
- The server repair candidate reuses the same action verify forward by averaging reconstructed endpoints across tau. A repair attempt adds one extra action verify forward only when original verify rejects.
- Launcher path was checked and updated so the experiment script uses this V7 code root, not the older worldlatent copy.

## Tests Run

- python3 -m pytest -q tests/test_specverify_client_policy.py::test_risk_router_repairs_action_reject_before_teacher_fallback tests/test_specverify_client_policy.py::test_risk_router_falls_back_when_repaired_action_still_rejects
- python3 -m pytest -q tests/test_specverify_client_policy.py tests/test_specverify.py
- python3 -m py_compile evaluation/robotwin/specverify_client_policy.py evaluation/robotwin/eval_polict_client_openpi.py wan_va/wan_va_server.py scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py

## Decision

Allowed to proceed to a low10 experiment with repair enabled as the only new capability relative to V6 worldlatent.
