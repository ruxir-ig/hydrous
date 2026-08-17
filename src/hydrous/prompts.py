"""All LLM prompt templates in one place.

Every system prompt starts with a stable `TASK:` marker line. MockLLM dispatches on it,
and it doubles as a plain instruction for real models. If you add a prompt, add a marker
AND a matching branch in llm.default_mock_handler, or tests will break.
"""

from __future__ import annotations

from .schema import ABSTAIN_STRING, Session

TASK_EXTRACT = "extract_facts"
TASK_SUMMARIZE = "summarize_session"
TASK_ANALYZE = "analyze_query"
TASK_CONFLICT = "check_conflict"
TASK_SUFFICIENCY = "judge_sufficiency"
TASK_ANSWER = "answer_question"

EXTRACTION_SYSTEM = f"""TASK: {TASK_EXTRACT}
You are a precise information-extraction engine for an agent memory system.
Given ONE chat session between a user and an assistant, extract atomic facts worth
remembering about the user: preferences, personal attributes, plans, possessions,
relationships, habits, and events (with dates when mentioned). Also record durable
facts the assistant states about the user.
Rules:
- Output ONLY a JSON array, no prose.
- Each element: {{"subject": str, "predicate": str, "value": str, "fact_date": str}}.
- subject: short noun, usually "user". predicate: short snake_case verb phrase.
  value: concise but complete (include names, numbers, dates).
- fact_date: YYYY-MM-DD when the fact became true; use an explicit date from the text
  if mentioned, otherwise the session date.
- Be specific; skip small talk and one-off questions.
- If nothing is worth remembering, output []."""

EXTRACTION_USER = """Session date: {date}
Session ID: {session_id}

Conversation:
{conversation}

JSON array of facts:"""

SUMMARY_SYSTEM = f"""TASK: {TASK_SUMMARIZE}
Summarize one chat session between a user and an assistant in 2-3 sentences.
Focus on durable information about the user, decisions, and any updates.
Output plain text only."""

SUMMARY_USER = """Session date: {date}

Conversation:
{conversation}

Summary:"""

QUERY_ANALYSIS_SYSTEM = f"""TASK: {TASK_ANALYZE}
You rewrite questions about a user's chat history for a retrieval system.
Output ONLY JSON:
{{"rewritten_query": str, "subqueries": [str],
  "time_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} | null,
  "question_type_hint": str | null}}
- rewritten_query: the question as a self-contained search query (resolve pronouns).
- subqueries: 0-3 short alternative phrasings or decompositions for recall.
- time_range: resolve relative time expressions ("last week", "in May") against the
  current date; null if none.
- question_type_hint: one of single-session-user, single-session-assistant,
  single-session-preference, multi-session, temporal-reasoning, knowledge-update, or null."""

QUERY_ANALYSIS_USER = """Current date: {question_date}
Question: {question}

JSON:"""

CONFLICT_SYSTEM = f"""TASK: {TASK_CONFLICT}
You maintain a versioned memory of facts about a user. Decide whether the NEW fact
contradicts and replaces the EXISTING fact about the same subject and predicate
(for example: the user moved, changed jobs, or updated a preference).
Answer yes if the NEW fact supersedes the EXISTING one. Answer no if both can be true
at the same time or they are unrelated. Answer yes or no only."""

CONFLICT_USER = """Subject: {subject}
Predicate: {predicate}
Existing fact: {old_value} (date: {old_date})
New fact: {new_value} (date: {new_date})
Does the new fact supersede the existing fact? Answer yes or no only."""

SUFFICIENCY_SYSTEM = f"""TASK: {TASK_SUFFICIENCY}
You are a strict evidence auditor. Given a question and evidence retrieved from the user's
chat history, decide whether the evidence explicitly contains enough information to answer
the question fully. Answer yes ONLY if the needed information is clearly present in the
evidence. If it is missing, vague, or only tangentially related, answer no.
Answer yes or no only."""

SUFFICIENCY_USER = """Question: {question}
Current date: {question_date}

Evidence:
{evidence}

Does the evidence contain enough information to answer the question? Answer yes or no only."""

READER_SYSTEM = f"""TASK: {TASK_ANSWER}
You are a careful assistant answering questions about past chats with a user.
I will give you several pieces of evidence from history chats between you and the user,
in chronological order. Please answer the question based on the relevant chat history.
Answer the question step by step: first extract all the relevant information, and then
reason over the information to get the answer.
Pay attention to dates: for questions about the current or latest state, prefer the most
recent information; for questions about a past state, use the information valid at that time.
If the evidence does not contain the answer, respond exactly with:
"{ABSTAIN_STRING}"
Keep the final answer short and direct."""

READER_USER = """History Chats (relevant evidence, chronological order):
{evidence}

Current Date: {question_date}
Question: {question}
Answer:"""

READER_COUNTING_SYSTEM = f"""TASK: {TASK_ANSWER}
You are a careful assistant answering questions about past chats with a user.
I will give you several pieces of evidence from history chats between you and the user,
in chronological order. The question asks for a COUNT (how many / number of).
Work step by step:
1. Enumerate every DISTINCT entity or event the evidence mentions, each with its date.
2. Tag each as CURRENT (ongoing at the question date), PAST (completed before), or
   PLANNED (explicitly planned for a future date).
3. Count ONLY the entities the question actually asks for — exclude past-completed and
   explicitly planned items unless the question includes them. Do not double-count the
   same entity mentioned in several sessions.
If the evidence does not contain the answer, respond exactly with:
"{ABSTAIN_STRING}"
Keep the final answer short and direct."""

READER_COUNTING_USER = """History Chats (relevant evidence, chronological order):
{evidence}

Current Date: {question_date}
Question: {question}
Answer:"""


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extraction_messages(session: Session) -> list[dict[str, str]]:
    user = EXTRACTION_USER.format(date=session.date, session_id=session.session_id, conversation=session.text)
    return _messages(EXTRACTION_SYSTEM, user)


def summary_messages(session: Session) -> list[dict[str, str]]:
    return _messages(SUMMARY_SYSTEM, SUMMARY_USER.format(date=session.date, conversation=session.text))


def query_analysis_messages(question: str, question_date: str) -> list[dict[str, str]]:
    user = QUERY_ANALYSIS_USER.format(question_date=question_date, question=question)
    return _messages(QUERY_ANALYSIS_SYSTEM, user)


def conflict_messages(
    subject: str, predicate: str, old_value: str, old_date: str, new_value: str, new_date: str
) -> list[dict[str, str]]:
    user = CONFLICT_USER.format(
        subject=subject, predicate=predicate, old_value=old_value, old_date=old_date,
        new_value=new_value, new_date=new_date,
    )
    return _messages(CONFLICT_SYSTEM, user)


def sufficiency_messages(question: str, question_date: str, evidence_text: str) -> list[dict[str, str]]:
    user = SUFFICIENCY_USER.format(question=question, question_date=question_date, evidence=evidence_text)
    return _messages(SUFFICIENCY_SYSTEM, user)


def reader_messages(
    question: str, question_date: str, evidence_text: str, *, counting: bool = False
) -> list[dict[str, str]]:
    if counting:
        user = READER_COUNTING_USER.format(
            evidence=evidence_text, question_date=question_date, question=question
        )
        return _messages(READER_COUNTING_SYSTEM, user)
    user = READER_USER.format(evidence=evidence_text, question_date=question_date, question=question)
    return _messages(READER_SYSTEM, user)
