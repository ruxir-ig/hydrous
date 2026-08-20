"""Native HTTP adapter for the local AGPL HydraDB OSS graph engine.

This deliberately does not import ``hydradb-sdk``. The OSS node exposes an
OpenCypher HTTP API, which is the source of truth for this adapter.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..schema import Fact, Session
from ..text_utils import compare_dates
from . import queries
from .base import GraphPath, MemoryGraph

log = logging.getLogger(__name__)


class HydraDBOSSGraph(MemoryGraph):
    def __init__(
        self,
        token: str,
        *,
        base_url: str = "http://127.0.0.1:8443",
        graph_id: str = "default",
        namespace: str = "default",
        cell_id: str = "cell-0",
        timeout: float = 15.0,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._graph_id = graph_id
        self._namespace = namespace
        self._cell_id = cell_id
        self._timeout = timeout
        self._last_session: dict[str, tuple[str, str]] = {}

    @classmethod
    def from_token_file(cls, token_file: str | Path, **kwargs: Any) -> HydraDBOSSGraph:
        path = Path(token_file)
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"HydraDB OSS token file is unreadable: {path}") from exc
        if not token:
            raise RuntimeError(f"HydraDB OSS token file is empty: {path}")
        return cls(token, **kwargs)

    def _request(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        payload = {"cell_id": self._cell_id, "query": cypher}
        if parameters:
            payload["parameters"] = {key: value for key, value in parameters.items() if value is not None}
        request = Request(
            f"{self._base_url}/v1/graphs/{self._graph_id}/query",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._token}",
                "X-Graph-Namespace": self._namespace,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HydraDB OSS query failed ({exc.code}): {detail[:500]}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"HydraDB OSS is unreachable at {self._base_url}; start `docker compose up -d hydradb`"
            ) from exc
        columns = body.get("columns", [])
        return [
            {column: self._decode(value) for column, value in zip(columns, row, strict=False)}
            for row in body.get("rows", [])
        ]

    @staticmethod
    def _decode(value: Any) -> Any:
        if not isinstance(value, dict) or "type" not in value:
            return value
        kind = value["type"]
        raw = value.get("value")
        if kind == "null":
            return None
        if kind == "list":
            return [HydraDBOSSGraph._decode(v) for v in raw or []]
        return raw

    async def _run(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._request, cypher, parameters)

    @staticmethod
    def _storage_id(value: str) -> int:
        """Map a stable Hydrous key to HydraDB OSS's integer identity space."""
        number = int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")
        return max(number & ((1 << 63) - 1), 1)

    @classmethod
    def _fact_params(cls, question_id: str, fact: Fact) -> dict[str, Any]:
        subject_key = f"entity:{question_id}:{fact.subject}"
        value_key = f"entity:{question_id}:{fact.value}"
        fact_key = f"fact:{question_id}:{fact.id}"
        return {
            "question_id": question_id,
            "fact_id_value": fact.id,
            "fact_vertex_id": cls._storage_id(fact_key),
            "fact_edge_id": cls._storage_id(f"predicate:{fact_key}"),
            "subject_key": cls._storage_id(subject_key),
            "value_key": cls._storage_id(value_key),
            "subject": fact.subject,
            "predicate": fact.predicate,
            "value": fact.value,
            "fact_date": fact.fact_date,
            "session_id": fact.session_id,
            "valid_from": fact.valid_from,
            "valid_to": fact.valid_to,
            "recorded_at": fact.recorded_at,
            "superseded_by": fact.superseded_by,
            "quality_score": fact.quality_score,
        }

    @staticmethod
    def _fact(row: dict[str, Any], question_id: str) -> Fact:
        return Fact(
            id=str(row.get("fact_id") or ""), question_id=question_id,
            subject=str(row.get("subject") or ""), predicate=str(row.get("predicate") or ""),
            value=str(row.get("value") or ""), fact_date=str(row.get("fact_date") or ""),
            session_id=str(row.get("session_id") or ""), valid_from=str(row.get("valid_from") or ""),
            valid_to=row.get("valid_to"), recorded_at=str(row.get("recorded_at") or ""),
            superseded_by=row.get("superseded_by"), quality_score=row.get("quality_score"),
        )

    async def put_session(self, question_id: str, session: Session) -> None:
        session_key = self._storage_id(f"session:{question_id}:{session.session_id}")
        params = {
            "question_id": question_id, "session_id": session.session_id,
            "session_key": session_key, "date": session.date,
            "recorded_at": session.recorded_at,
        }
        if session.summary is not None:
            params["summary"] = session.summary
        marker_id = self._storage_id(f"session-marker:{question_id}:{session.session_id}")
        await self._run(queries.merge_session(), {"session_key": session_key, "marker_id": marker_id})
        await self._run(queries.set_marker(), {"marker_id": marker_id, "question_id": question_id})
        await self._run(queries.set_session(has_summary=session.summary is not None), params)
        previous = self._last_session.get(question_id)
        if previous and previous[0] <= session.date and previous[1] != session_key:
            await self._run(queries.link_sessions(), {
                "previous_key": previous[1], "session_key": session_key,
                "edge_id": self._storage_id(f"next:{question_id}:{previous[1]}:{session_key}"),
            })
        self._last_session[question_id] = (session.date, session_key)

    async def put_facts(self, question_id: str, facts: list[Fact]) -> None:
        for fact in facts:
            params = self._fact_params(question_id, fact)
            await self._run(queries.merge_predicate(), params)
            await self._run(queries.set_entity("subject_key"), params)
            await self._run(queries.set_entity("value_key"), params)
            await self._run(queries.link_fact_session(), {
                "fact_vertex_id": params["fact_vertex_id"],
                "session_key": self._storage_id(f"session:{question_id}:{fact.session_id}"),
                "from_edge_id": self._storage_id(f"from:{question_id}:{fact.id}"),
            })
            await self._run(
                queries.set_fact(
                    has_valid_to=fact.valid_to is not None,
                    has_superseded_by=fact.superseded_by is not None,
                    has_quality_score=fact.quality_score is not None,
                ),
                params,
            )
            await self._run(
                queries.set_predicate(
                    has_valid_to=fact.valid_to is not None,
                    has_superseded_by=fact.superseded_by is not None,
                    has_quality_score=fact.quality_score is not None,
                ),
                params,
            )
            if fact.superseded_by:
                await self._run(
                    queries.link_superseded(),
                    {
                        **params,
                        "replacement_vertex_id": self._storage_id(
                            f"fact:{question_id}:{fact.superseded_by}"
                        ),
                        "superseded_edge_id": self._storage_id(
                            f"superseded:{question_id}:{fact.id}:{fact.superseded_by}"
                        ),
                    },
                )

    async def update_fact(self, fact: Fact) -> None:
        params = self._fact_params(fact.question_id, fact)
        params["superseded_by_value"] = fact.superseded_by
        params["replacement_vertex_id"] = (
            self._storage_id(f"fact:{fact.question_id}:{fact.superseded_by}")
            if fact.superseded_by else None
        )
        params["superseded_edge_id"] = self._storage_id(
            f"superseded:{fact.question_id}:{fact.id}:{fact.superseded_by}"
        )
        await self._run(
            queries.update_fact(
                has_valid_to=fact.valid_to is not None,
                has_superseded_by=fact.superseded_by is not None,
            ),
            {k: v for k, v in params.items() if v is not None},
        )
        await self._run(
            queries.update_predicate(
                has_valid_to=fact.valid_to is not None,
                has_superseded_by=fact.superseded_by is not None,
            ),
            {k: v for k, v in params.items() if v is not None},
        )
        if fact.superseded_by:
            await self._run(queries.link_superseded(), params)

    async def as_of(self, question_id: str, reference_date: str | None = None) -> list[Fact]:
        params = {"question_id": question_id}
        if reference_date is not None:
            params["reference_date"] = reference_date
        rows = await self._run(queries.as_of(), params)
        return [
            fact for fact in (self._fact(row, question_id) for row in rows)
            if fact.valid_as_of(reference_date)
        ]

    async def supersession_chain(self, question_id: str, subject: str, predicate: str) -> list[Fact]:
        candidates = [
            self._fact(row, question_id)
            for row in await self._run(queries.as_of(), {"question_id": question_id})
            if row.get("subject") == subject and row.get("predicate") == predicate
        ]
        if not candidates:
            return []
        target_rows = await self._run(queries.supersession_targets(), {"question_id": question_id})
        target_ids = {str(row.get("replacement_id")) for row in target_rows}
        roots = [
            fact for fact in candidates
            if str(self._storage_id(f"fact:{question_id}:{fact.id}")) not in target_ids
        ] or candidates

        best: list[Fact] = []
        for root in sorted(roots, key=lambda fact: fact.valid_from):
            rows = await self._run(
                queries.supersession_chain(),
                {
                    "question_id": question_id,
                    "start_fact_key": self._storage_id(f"fact:{question_id}:{root.id}"),
                },
            )
            chain = [root, *(self._fact(row, question_id) for row in rows)]
            chain = list({fact.id: fact for fact in chain}.values())
            chain.sort(key=lambda fact: fact.valid_from)
            if len(chain) > len(best):
                best = chain
        return best

    async def multi_hop(
        self, question_id: str, start_subject: str, predicates: list[str], reference_date: str | None = None
    ) -> list[GraphPath]:
        if not predicates or len(predicates) > 4:
            raise ValueError("multi_hop requires between 1 and 4 predicates")
        params: dict[str, Any] = {
            "question_id": question_id,
            "start_key": self._storage_id(f"entity:{question_id}:{start_subject}"),
            "reference_date": reference_date,
        }
        for i, predicate in enumerate(predicates):
            params[f"predicate_{i}"] = predicate
        cypher = queries.multi_hop(hops=len(predicates))
        rows = await self._run(cypher, params)
        paths = []
        for row in rows:
            valid = True
            for i in range(len(predicates)):
                start = str(row.get(f"valid_from_{i}") or "")
                end = row.get(f"valid_to_{i}")
                if reference_date is None:
                    valid = valid and end is None
                else:
                    valid = valid and compare_dates(start, reference_date) <= 0
                    valid = valid and (end is None or compare_dates(reference_date, str(end)) < 0)
            if not valid:
                continue
            paths.append(GraphPath(
                nodes=tuple(str(row.get(f"node_{i}") or "") for i in range(len(predicates) + 1)),
                predicates=tuple(predicates),
                facts=tuple(str(row.get(f"fact_{i}") or "") for i in range(len(predicates))),
                cypher=cypher,
            ))
        return paths

    async def delete_memory(self, question_id: str, memory_id: str, *, kind: str) -> None:
        if kind == "fact":
            fact_key = self._storage_id(f"fact:{question_id}:{memory_id}")
            await self._run(
                queries.delete_fact_edge(),
                {"fact_edge_id": self._storage_id(f"predicate:fact:{question_id}:{memory_id}")},
            )
            await self._run(queries.delete_fact(), {"fact_vertex_id": fact_key})
            return
        if kind == "session":
            session_key = self._storage_id(f"session:{question_id}:{memory_id}")
            marker_id = self._storage_id(f"session-marker:{question_id}:{memory_id}")
            await self._run(queries.delete_session(), {"session_key": session_key})
            await self._run(queries.delete_session_marker(), {"marker_id": marker_id})
            self._last_session.pop(question_id, None)
            return
        raise ValueError(f"unknown graph memory kind: {kind}")

    async def delete_question(self, question_id: str) -> None:
        for label in ("Entity", "Fact", "Session", "SessionMarker"):
            await self._run(queries.delete_scope(label), {"question_id": question_id})
        self._last_session.pop(question_id, None)

    def close(self) -> None:
        return None
