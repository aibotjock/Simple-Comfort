# Legacy LTX-2.5 backend graphs (superseded 2026-09-07)

These four API-format templates were the app's first LTX-2.5 backend
(`ltx25_t2v.json`, `ltx25_i2v.json`, `ltx25_t2v_2stage.json`,
`ltx25_i2v_2stage.json`). They were hand-derived from the official
LTX-2.5 distilled templates and drove the `ltx25` engine until the
AAN candidate pack was ingested.

**Superseded by:** `workflows/aan_ltx25/` — the ten byte-faithful
Applied AI News LTX-2.5 workflows (verified against the pack's
`SHA256_MANIFEST.json`), now the LTX-2.5 backend set used by
`adapters/ltx25.py`. The pack graphs cover everything the legacy set did
(t2v/i2v via in-graph toggles, single-stage + two-stage quality tiers)
plus joint audio/video end-to-end, the in-graph Gemma prompt enhancer,
and the pixel-spatial IC-LoRA finishing profile.

Files are kept here as loadable reference only — nothing in the app
discovers them anymore (adapters, `tools/check_models.py` and
`validate_templates.py` all point at the pack). They are not maintained.
