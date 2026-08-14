# Hydrous

Agent memory layer for cross-session continuity — "Hack with Hydra" hackathon project.

**Read [ARCHITECTURE.md](ARCHITECTURE.md) — it is the single living doc**: problem, design,
algorithms, eval protocol, how to run, current status, and handoff notes.

## Quickstart

```bash
uv sync
cp .env.example .env   # add your API key
uv run hydra-mem download-data
uv run hydra-mem eval --limit 10          # real run (needs API key in .env)
uv run hydra-mem eval --limit 10 --mock   # deterministic dry-run, no key needed
uv run pytest                             # all tests pass with no API key
```
