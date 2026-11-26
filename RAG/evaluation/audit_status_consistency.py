import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = ROOT / "RAG" / "data" / "verified_knowledge.jsonl"
REPORT_CSV = ROOT / "RAG" / "evaluation" / "status_consistency_audit.csv"

PUBLIC_GAP_RE = re.compile(r"\b(not publicly|does not publicly|don't publish|doesn't publish|not publish|not specified|not stated|not documented|no official)\b", re.I)
ROUTING_RE = re.compile(r"\b(contact|advisor|adviser|branch|centre de relation client|support|customer service|bank intervention)\b", re.I)
PRIVATE_DATA_RE = re.compile(r"\b(account|transaction|card|login|application|device|personal|specific situation|status)\b", re.I)
UNCERTAINTY_RE = re.compile(r"\b(however|not confirmed|not specified|not stated|not published|verify|confirm|could not|we could not|depends)\b", re.I)
CONTRADICTION_RE = re.compile(r"\bremoved|unsupported|not confirmed|could not confirm|does not appear|not present|cleared rather than left misleading\b", re.I)


def recommendation_for(document, problem):
    status = document.get("verification_status")
    product = document.get("product") or "the product"
    question_type = document.get("question_type") or "detail"
    if status == "official_information_not_public":
        return (
            f"State that Atlas Demo Bank does not publicly document the requested {question_type} for {product}; "
            "then include only confirmed context and route to an advisor or branch."
        )
    if status == "human_intervention":
        return (
            "Keep only safe general guidance and tell the customer the issue requires Atlas Demo Bank support, "
            "without diagnosing account, card, login, transaction, or application state."
        )
    if status == "partially_verified":
        return (
            "Separate confirmed facts from unconfirmed details, and add a clear qualification for the uncertain portion."
        )
    return f"Review because: {problem}"


def find_problem(document):
    status = document.get("verification_status")
    answer = " ".join(str(document.get("answer") or document.get("content") or "").split())
    notes = " ".join(str(document.get("verification_notes") or "").split())

    if status == "official_information_not_public":
        if not PUBLIC_GAP_RE.search(answer):
            return "Answer does not clearly say the requested detail is not publicly documented."
        if not ROUTING_RE.search(answer):
            return "Answer does not route the customer to an official bank channel."

    if status == "human_intervention":
        if not ROUTING_RE.search(answer):
            return "Answer does not route to human/bank support."
        if not PRIVATE_DATA_RE.search(answer):
            return "Answer may not clearly explain that private/customer-specific handling is required."

    if status == "partially_verified":
        if not UNCERTAINTY_RE.search(answer):
            return "Answer is partially verified but lacks an explicit qualification."

    if CONTRADICTION_RE.search(notes) and any(
        phrase in answer.lower()
        for phrase in ("by sms", "3d secure", "online purchases", "cash advance", "last 8", "last-8")
    ):
        return "Verification notes flag unsupported/removed claims that may still appear in the answer."

    return None


def main():
    findings = []
    with INPUT_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            document = json.loads(line)
            if document.get("verification_status") not in {
                "official_information_not_public",
                "human_intervention",
                "partially_verified",
            }:
                continue
            problem = find_problem(document)
            if not problem:
                continue
            findings.append({
                "question_id": document.get("id"),
                "question": document.get("question"),
                "status": document.get("verification_status"),
                "problem": problem,
                "current answer": document.get("answer") or document.get("content"),
                "recommended correction": recommendation_for(document, problem),
            })

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "question_id",
            "question",
            "status",
            "problem",
            "current answer",
            "recommended correction",
        ])
        writer.writeheader()
        writer.writerows(findings)

    print(f"Rows audited: {sum(1 for _ in INPUT_JSONL.open(encoding='utf-8') if _.strip())}")
    print(f"Findings: {len(findings)}")
    print(f"Report: {REPORT_CSV}")
    for finding in findings[:20]:
        print(json.dumps(finding, ensure_ascii=False))


if __name__ == "__main__":
    main()
