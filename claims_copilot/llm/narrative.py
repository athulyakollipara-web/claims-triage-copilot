"""The investigator-facing explanation.

The decision (fast-track / review / refer) is made by governed rules over the
model score and evidence, never by the LLM. The LLM only explains it. It
receives numbered facts ([D1] risk drivers, [M1] mitigating factors, [E1]
evidence) and must cite them; ``check_faithfulness`` rejects narratives that cite
facts that don't exist, leave claims uncited, or introduce new numbers.
"""
from __future__ import annotations

import re

ACTION_LABELS = {"refer_to_siu": "refer to the Special Investigations Unit (SIU)", "enhanced_review": "enhanced review",
                 "standard_review": "standard review", "fast_track": "fast-track for payment"}


def _doc(name: str) -> str:
    return name.replace("_", " ").replace(" id", " ID")

SYSTEM = """You write short case summaries for insurance claims investigators.
Use ONLY the numbered facts provided. Every sentence must cite at least one fact id in square brackets, e.g. [D1] or [E2].
Do not add numbers, names or facts that are not in the list. Do not speculate about guilt; describe risk and next steps.
Write 3-5 sentences. End with the recommended action and the next steps."""


def facts_from(case: dict) -> dict[str, str]:
    facts = {}
    for i, d in enumerate(case["explanation"]["risk_drivers"], 1):
        facts[f"D{i}"] = d["description"]
    for i, d in enumerate(case["explanation"]["mitigating"], 1):
        facts[f"M{i}"] = d["description"]
    ev = case["evidence"]
    idx = 1
    for doc in ev["missing_documents"]:
        facts[f"E{idx}"] = f"missing document: {_doc(doc)}"
        idx += 1
    for flag in ev["note_flags"]:
        facts[f"E{idx}"] = f"adjuster note flag: {flag.replace('_', ' ')}"
        idx += 1
    facts["S1"] = f"model risk score {case['score']:.2f} ({case['recommendation']['tier']} risk tier)"
    return facts


def prompt_for(case: dict, facts: dict[str, str]) -> str:
    lines = [f"[{k}] {v}" for k, v in facts.items()]
    rec = case["recommendation"]
    return ("FACTS:\n" + "\n".join(lines) +
            f"\n\nRECOMMENDED ACTION: {rec['action']}\nNEXT STEPS: " + "; ".join(rec["next_steps"]))


def template_narrative(case: dict, facts: dict[str, str]) -> str:
    """Deterministic narrative used offline and as the fallback if a model's output fails the check."""
    rec = case["recommendation"]
    drivers = [k for k in facts if k.startswith("D")][:3]
    parts = [f"This claim scored in the {rec['tier']} risk tier [S1]."]
    if drivers:
        parts.append("The main risk factors are: " + "; ".join(f"{facts[k]} [{k}]" for k in drivers) + ".")
    mitig = [k for k in facts if k.startswith("M")][:1]
    if mitig:
        parts.append(f"In its favour, {facts[mitig[0]]} [{mitig[0]}].")
    ev = [k for k in facts if k.startswith("E")]
    if ev:
        parts.append("The file shows " + "; ".join(f"{facts[k]} [{k}]" for k in ev[:3]) + ".")
    cite = (drivers or ev or ["S1"])[0]
    parts.append(f"Recommended action: {ACTION_LABELS.get(rec['action'], rec['action'])}; next steps: {'; '.join(rec['next_steps'])} [{cite}].")
    return " ".join(parts)


def check_faithfulness(text: str, facts: dict[str, str]) -> dict:
    cited = set(re.findall(r"\[([A-Z]\d+)\]", text))
    unknown = sorted(cited - set(facts))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s) > 3]
    uncited = [s for s in sentences if not re.search(r"\[[A-Z]\d+\]", s)]
    fact_text = " ".join(facts.values())
    allowed_numbers = set(re.findall(r"\d[\d,]*\.?\d*", fact_text))
    numbers = re.findall(r"(?<![A-Z])\d[\d,]*\.?\d*", re.sub(r"\[[A-Z]\d+\]", "", text))
    new_numbers = sorted({n.rstrip(".,") for n in numbers if n.rstrip(".,") not in allowed_numbers})
    ok = not unknown and not uncited and not new_numbers
    return {"faithful": ok, "unknown_citations": unknown, "uncited_sentences": uncited, "new_numbers": new_numbers}


class BedrockNarrator:
    def __init__(self, model_ids: list[str], region: str):
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_ids = model_ids

    def write(self, case: dict, facts: dict[str, str]) -> tuple[str, str]:
        from botocore.exceptions import ClientError

        last = None
        for model_id in self.model_ids:
            try:
                resp = self.client.converse(
                    modelId=model_id, system=[{"text": SYSTEM}],
                    messages=[{"role": "user", "content": [{"text": prompt_for(case, facts)}]}],
                    inferenceConfig={"maxTokens": 400, "temperature": 0},
                )
                return "".join(b.get("text", "") for b in resp["output"]["message"]["content"]).strip(), model_id
            except ClientError as e:
                last = e
        raise RuntimeError(f"All narrative models failed: {last}")


class TemplateNarrator:
    def write(self, case: dict, facts: dict[str, str]) -> tuple[str, str]:
        return template_narrative(case, facts), "template"
