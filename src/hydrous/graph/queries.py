"""Private OpenCypher builders for the HydraDB OSS adapter."""

from __future__ import annotations


def merge_session() -> str:
    # OSS HydraDB's MERGE executor currently accepts one-hop edge patterns,
    # not node-only MERGE. The marker is an internal anchor for raw sessions
    # that have no fact edge yet.
    return (
        "MERGE (s:Session {id: $session_key})-"
        "[:SESSION_MARKER {id: $marker_id}]->"
        "(marker:SessionMarker {id: $marker_id})"
    )


def set_marker() -> str:
    return "MATCH (marker:SessionMarker {id: $marker_id}) SET marker.question_id = $question_id"


def set_session(*, has_summary: bool) -> str:
    fields = (
        "s.session_id = $session_id, s.question_id = $question_id, "
        "s.date = $date, s.recorded_at = $recorded_at"
    )
    if has_summary:
        fields += ", s.summary = $summary"
        cleanup = ""
    else:
        cleanup = " REMOVE s.summary"
    return f"MATCH (s:Session {{id: $session_key}}) SET {fields}{cleanup}"


def link_sessions() -> str:
    return """
    MATCH (previous:Session {id: $previous_key}), (current:Session {id: $session_key})
    MERGE (previous)-[:NEXT_SESSION {id: $edge_id}]->(current)
    """.strip()


def set_entity(param: str) -> str:
    name = param.removesuffix("_key")
    return (
        f"MATCH (n:Entity {{id: ${param}}}) "
        f"SET n.name = ${name}, n.question_id = $question_id"
    )


def set_fact(*, has_valid_to: bool, has_superseded_by: bool, has_quality_score: bool) -> str:
    fields = [
        "fact.fact_id = $fact_id_value", "fact.question_id = $question_id", "fact.subject = $subject",
        "fact.predicate = $predicate", "fact.value = $value", "fact.fact_date = $fact_date",
        "fact.session_id = $session_id", "fact.valid_from = $valid_from", "fact.recorded_at = $recorded_at",
    ]
    cleanup: list[str] = []
    for name, enabled in (
        ("valid_to", has_valid_to), ("superseded_by", has_superseded_by), ("quality_score", has_quality_score)
    ):
        if enabled:
            fields.append(f"fact.{name} = ${name}")
        else:
            cleanup.append(f"fact.{name}")
    query = f"MATCH (fact:Fact {{id: $fact_vertex_id}}) SET {', '.join(fields)}"
    if cleanup:
        query += " REMOVE " + ", ".join(cleanup)
    return query


def merge_predicate() -> str:
    return (
        "MERGE (subject:Entity {id: $subject_key})-"
        "[r:PREDICATE {id: $fact_edge_id}]->(value:Entity {id: $value_key})"
    )


def set_predicate(*, has_valid_to: bool, has_superseded_by: bool, has_quality_score: bool) -> str:
    fields = [
        "r.fact_id = $fact_id_value", "r.question_id = $question_id", "r.name = $predicate",
        "r.fact_date = $fact_date", "r.valid_from = $valid_from", "r.recorded_at = $recorded_at",
        "r.session_id = $session_id",
    ]
    cleanup: list[str] = []
    for name, enabled in (
        ("valid_to", has_valid_to), ("superseded_by", has_superseded_by), ("quality_score", has_quality_score)
    ):
        if enabled:
            fields.append(f"r.{name} = ${name}")
        else:
            cleanup.append(f"r.{name}")
    query = (
        "MATCH (subject:Entity)-[r:PREDICATE {id: $fact_edge_id}]->(value:Entity) "
        f"SET {', '.join(fields)}"
    )
    if cleanup:
        query += " REMOVE " + ", ".join(cleanup)
    return query


def link_fact_session() -> str:
    return (
        "MERGE (fact:Fact {id: $fact_vertex_id})-"
        "[:FROM_SESSION {id: $from_edge_id}]->(session:Session {id: $session_key})"
    )


def update_fact(*, has_valid_to: bool, has_superseded_by: bool) -> str:
    valid = "SET fact.valid_to = $valid_to" if has_valid_to else "REMOVE fact.valid_to"
    superseded = (
        "SET fact.superseded_by = $superseded_by_value"
        if has_superseded_by
        else "REMOVE fact.superseded_by"
    )
    return f"""
    MATCH (fact:Fact {{id: $fact_vertex_id}})
    {valid}
    {superseded}
    """.strip()


def update_predicate(*, has_valid_to: bool, has_superseded_by: bool) -> str:
    valid = "SET r.valid_to = $valid_to" if has_valid_to else "REMOVE r.valid_to"
    superseded = (
        "SET r.superseded_by = $superseded_by_value"
        if has_superseded_by
        else "REMOVE r.superseded_by"
    )
    return f"""
    MATCH (subject:Entity)-[r:PREDICATE {{id: $fact_edge_id}}]->(value:Entity)
    {valid}
    {superseded}
    """.strip()


def link_superseded() -> str:
    return (
        "MERGE (fact:Fact {id: $fact_vertex_id})-"
        "[:SUPERSEDED_BY {id: $superseded_edge_id}]->"
        "(replacement:Fact {id: $replacement_vertex_id})"
    )


def as_of() -> str:
    # Interval semantics are applied by the adapter because the OSS parser's
    # NULL predicate support is narrower than its ordinary comparisons.
    return """
    MATCH (subject:Entity)-[r:PREDICATE]->(value:Entity)
    WHERE r.question_id = $question_id
    RETURN r.fact_id AS fact_id, subject.name AS subject, r.name AS predicate,
           value.name AS value, r.fact_date AS fact_date, r.session_id AS session_id,
           r.valid_from AS valid_from, r.valid_to AS valid_to,
           r.recorded_at AS recorded_at, r.superseded_by AS superseded_by,
           r.quality_score AS quality_score
    ORDER BY r.valid_from
    """.strip()


def supersession_chain() -> str:
    return """
    MATCH (start:Fact {id: $start_fact_key})-[:SUPERSEDED_BY*1..4]->(fact:Fact)
    WHERE fact.question_id = $question_id
    RETURN fact.fact_id AS fact_id, fact.subject AS subject, fact.predicate AS predicate,
           fact.value AS value, fact.fact_date AS fact_date, fact.session_id AS session_id,
           fact.valid_from AS valid_from, fact.valid_to AS valid_to,
           fact.recorded_at AS recorded_at, fact.superseded_by AS superseded_by,
           fact.quality_score AS quality_score
    ORDER BY fact.valid_from
    """.strip()


def supersession_targets() -> str:
    return """
    MATCH (old:Fact)-[:SUPERSEDED_BY]->(replacement:Fact)
    WHERE old.question_id = $question_id
    RETURN replacement.id AS replacement_id
    """.strip()


def multi_hop(*, hops: int) -> str:
    nodes = [f"(n{i}:Entity)" for i in range(hops + 1)]
    pattern = nodes[0]
    for i in range(hops):
        pattern += f"-[r{i}:PREDICATE]->{nodes[i + 1]}"
    predicates = ["n0.id = $start_key"]
    for i in range(hops):
        predicates.append(f"r{i}.question_id = $question_id AND r{i}.name = $predicate_{i}")
    returns = [f"n{i}.name AS node_{i}" for i in range(hops + 1)]
    returns.extend(f"r{i}.fact_id AS fact_{i}" for i in range(hops))
    returns.extend(f"r{i}.valid_from AS valid_from_{i}" for i in range(hops))
    returns.extend(f"r{i}.valid_to AS valid_to_{i}" for i in range(hops))
    return f"MATCH {pattern} WHERE {' AND '.join(predicates)} RETURN {', '.join(returns)}"


def delete_scope(label: str) -> str:
    return f"MATCH (n:{label} {{question_id: $question_id}}) DETACH DELETE n"


def delete_fact_edge() -> str:
    return "MATCH (subject:Entity)-[r:PREDICATE {id: $fact_edge_id}]->(value:Entity) DELETE r"


def delete_fact() -> str:
    return "MATCH (fact:Fact {id: $fact_vertex_id}) DETACH DELETE fact"


def delete_session() -> str:
    return "MATCH (session:Session {id: $session_key}) DETACH DELETE session"


def delete_session_marker() -> str:
    return "MATCH (marker:SessionMarker {id: $marker_id}) DETACH DELETE marker"
