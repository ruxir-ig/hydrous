"""LongMemEval official judge prompts (verbatim) + judging loop.

Label rule: `'yes' in judge_response.lower()`. Judge model configurable (JUDGE_MODEL,
default gpt-4o-2024-08-06, temperature 0). Output: `<hypotheses>.log` JSONL, one record
per hypothesis with the judgment — mirroring the official evaluate_qa.py log.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from ..ingest.longmemeval import load_gold_index
from ..llm import LLMClient

log = logging.getLogger(__name__)

JUDGE_DEFAULT = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\nQuestion: {}\nCorrect Answer: {}\nModel Response: {}\nIs the model response correct? Answer yes or no only."

JUDGE_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. For your judgment, do not penalize the off-by-one errors for the number of days.\nQuestion: {}\nCorrect Answer: {}\nModel Response: {}\nIs the model response correct? Answer yes or no only."

JUDGE_KNOWLEDGE_UPDATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\nQuestion: {}\nCorrect Answer: {}\nModel Response: {}\nIs the model response correct? Answer yes or no only."

JUDGE_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\nQuestion: {}\nRubric: {}\nModel Response: {}\nIs the model response correct? Answer yes or no only."

JUDGE_ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\nQuestion: {}\nExplanation: {}\nModel Response: {}\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."


def select_template(question_id: str, question_type: str) -> str:
    """Abstention is detected via the `_abs` id suffix, not the question_type."""
    if question_id.endswith("_abs"):
        return JUDGE_ABSTENTION
    if question_type == "temporal-reasoning":
        return JUDGE_TEMPORAL
    if question_type == "knowledge-update":
        return JUDGE_KNOWLEDGE_UPDATE
    if question_type == "single-session-preference":
        return JUDGE_PREFERENCE
    return JUDGE_DEFAULT


async def judge_hypotheses(
    hypotheses_path: str | Path,
    dataset_path: str | Path,
    out_path: str | Path | None = None,
    *,
    llm: LLMClient,
    judge_model: str,
    max_concurrency: int = 8,
) -> list[dict]:
    """Judge every hypothesis against the gold answer; write + return log records."""
    gold = load_gold_index(dataset_path)
    hypotheses = [
        json.loads(line)
        for line in Path(hypotheses_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out = Path(out_path) if out_path else Path(str(hypotheses_path) + ".log")
    sem = asyncio.Semaphore(max_concurrency)

    async def judge_one(hyp: dict) -> dict:
        qid = hyp["question_id"]
        inst = gold.get(qid)
        if inst is None:
            log.warning("no gold instance for %s — skipped", qid)
            return {"question_id": qid, "question_type": None, "hypothesis": hyp["hypothesis"],
                    "judgment": None, "judge_response": "missing gold instance"}
        template = select_template(qid, inst["question_type"])
        prompt = template.format(inst["question"], inst["answer"], hyp["hypothesis"])
        async with sem:
            resp = await llm.chat(
                [{"role": "user", "content": prompt}], model=judge_model, temperature=0.0
            )
        return {
            "question_id": qid,
            "question_type": inst["question_type"],
            "hypothesis": hyp["hypothesis"],
            "judgment": "yes" in resp.lower(),
            "judge_response": resp,
        }

    records = list(await asyncio.gather(*[judge_one(h) for h in hypotheses]))
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records
