# zrok v2 Permanent Public Share (replaces Cloudflare quick tunnel)

**Date:** 2026-08-13
**Status:** Implemented in `api_server.py`, verified against zrok2 v2.0.4

## Goal

Replace the Cloudflare quick tunnel (`*.trycloudflare.com`, random per restart) with a
**zrok v2 reserved name** giving a permanent `https://dubstudio1.shares.zrok.io` URL that
auto-forwards `localhost:8002` (FastAPI + Gradio WebUI). Nothing else about the server changes.

## Decisions

| Decision | Value |
|---|---|
| Infrastructure | zrok.io SaaS (free tier) |
| Permanence | Reserved name (`create name`) |
| Cloudflare | Kept as `--share-backend cloudflared` fallback |
| Subdomain | `dubstudio1` (default `--share-name`) |
| Deployment | **Kaggle** (API server runs there, not this Windows dev box) |

## Verified zrok v2 command model (v2.0.4)

zrok v2 **removed** `zrok reserve` / `zrok share reserved`. The binary is `zrok2`.
Equivalent v2 flow, confirmed live:

```
zrok2 enable <token>                                            # one-time → ~/.zrok/environment.json
zrok2 create name dubstudio1                                    # reserve subdomain (409 = already reserved)
zrok2 share public http://localhost:8002 -n public:dubstudio1 --headless --force-local
                                                                # → https://dubstudio1.shares.zrok.io
```

- Name-selection format is `<namespaceToken>:<name>` (e.g. `public:dubstudio1`).
- Free-tier namespace token is `public` → `*.shares.zrok.io` (plural).
- `create name` is **not** idempotent — re-reserving returns `409 createShareNameConflict`,
  which the code treats as success.

## CLI surface (rank 0 only)

- `--share` — expose publicly (default backend zrok)
- `--share-name` (default `dubstudio1`) — reserved subdomain
- `--share-backend {zrok,cloudflared}` (default `zrok`)
- `--zrok-token` / `ZROK_ENABLE_TOKEN` — one-time enable token

## Provisioning flow (all non-fatal to local API)

1. Locate/download the `zrok2` binary → cache `./bin/zrok2`
2. If `~/.zrok/environment.json` missing → `zrok2 enable <token>`; no token → print myzrok.io instructions
3. `zrok2 create name <name>` (treat `409 conflict` as already-reserved success)
4. `zrok2 share public http://localhost:<port> -n public:<name> --headless --force-local`
   (background, logs to `logs/zrok-share.log`)
5. Print the known permanent URL (deterministic — no stdout parsing)

## Untouched

Multi-GPU `_forward_or_local` load balancing, async `job_id` paths, Gradio mount, port 8002,
`--fp16`/`--deepspeed`/`--accel`/`--cuda_kernel`, rank 1+ hidden worker API.

## Kaggle-specific note

Kaggle's filesystem is ephemeral: `~/.zrok/environment.json` (the environment identity) is
wiped between sessions. The reserved *name* lives server-side on the account, so it persists —
only the local enable needs redoing. On each fresh Kaggle session, pass a fresh
`--zrok-token` from https://myzrok.io (tokens are one-time), or persist `~/.zrok` via a
Kaggle dataset and symlink it to `~/.zrok` before launch.
