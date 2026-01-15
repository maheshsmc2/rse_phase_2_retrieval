# rse_phase_3_rerank (Phase 3)

Phase 3 adds:
- Reranking (Cross-Encoder)
- Confidence gating (ANSWER vs ABSTAIN)
- Decision trace saved as JSON
- CLI execution

## Important constraints
- Phase 1 is frozen
- Phase 2 exists as `rse_phase_2_retrieval`
- Phase 2 must NOT be modified
- Phase 3 imports Phase 2 and consumes Phase 2 top-k hits

## Install
From repo root:

```bash
pip install -r rse_phase_3_rerank/requirements.txt
