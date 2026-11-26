from __future__ import annotations

import re


TRANSACTION_RE = re.compile(r"\b(transfer|send|wire|pay|withdraw|deposit|change|close|open\s+an\s+account|block\s+my|freeze|unfreeze)\b", re.I)
TRANSACTION_INFO_RE = re.compile(r"\b(can|could|how|what|is|are|possible|work|works|procedure|steps|information|document|require)\b", re.I)
TRANSACTION_EXECUTION_RE = re.compile(r"\b(\d+(?:[.,]\d+)?\s*(?:tnd|dt|dinar|dinars)?|recipient|beneficiary|for me|from my account|to this|now|today)\b", re.I)
HUMAN_RE = re.compile(r"\b(loan rejected|rejected|approved|approval|fraud|investigation|frozen|restricted|restriction|appeal|bank decision|why was my loan)\b", re.I)
CUSTOMER_RE = re.compile(
    r"\b("
    r"products do i own|do i own|my current products|my products|my banking segment|my segment|my banking relationship|my profile|my balance|recommend(?:ed|ation)? for me|why do you recommend|"
    r"explain the top recommendation|summarize this customer|available context|customer's behavior|segment forecast|"
    r"which of my|which products do i currently own|show me my|my card details|my account information|my account details|status of my|currently blocked|"
    r"am i personally eligible|personally eligible for|based on my"
    r")\b",
    re.I,
)
IDENTITY_OVERRIDE_RE = re.compile(r"\b(pretend|impersonate|act as|ignore|override|customer\s+\d+|another customer|raw retrieved|hidden prompt)\b", re.I)
RECOMMENDATION_RE = re.compile(r"\b(recommend(?:ed|ation)?|fits my profile|best product for me|suggest.*for me|suit(?:s)? me|would be right for me)\b", re.I)
OTHER_CUSTOMER_RE = re.compile(r"\b(another customer|other customer|customer\s+\d+)\b", re.I)


def is_transaction_execution_request(message: str) -> bool:
    return bool(TRANSACTION_RE.search(message or "") and (TRANSACTION_EXECUTION_RE.search(message or "") or not TRANSACTION_INFO_RE.search(message or "")))


def is_other_customer_reference(message: str) -> bool:
    return bool(OTHER_CUSTOMER_RE.search(message or ""))


def is_identity_override_request(message: str) -> bool:
    return bool(IDENTITY_OVERRIDE_RE.search(message or ""))
