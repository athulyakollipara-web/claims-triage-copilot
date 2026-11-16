"""Tamper-evident audit log of every recommendation and human decision.

Each entry stores the SHA-256 of the previous entry, so editing or deleting any
past record breaks the chain and ``verify()`` reports where. Local mode writes
JSON lines; AWS mode writes the same records to DynamoDB (partition key
``claim_id``, sort key ``seq``).
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


def _hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class FileAuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, claim_id: str, event: str, actor: str, payload: dict) -> dict:
        with self._lock:
            entries = self._entries()
            entry = {
                "seq": len(entries) + 1, "ts": datetime.now(timezone.utc).isoformat(), "claim_id": claim_id,
                "event": event, "actor": actor, "payload": payload,
                "prev_hash": entries[-1]["hash"] if entries else "0" * 64,
            }
            entry["hash"] = _hash(entry)
            with self.path.open("a") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
            return entry

    def for_claim(self, claim_id: str) -> list[dict]:
        return [e for e in self._entries() if e["claim_id"] == claim_id]

    def verify(self) -> tuple[bool, int | None]:
        prev = "0" * 64
        for e in self._entries():
            if e["prev_hash"] != prev or _hash(e) != e["hash"]:
                return False, e["seq"]
            prev = e["hash"]
        return True, None


class DynamoAuditLog:
    """Same interface backed by DynamoDB. The chain is kept per claim."""

    def __init__(self, table: str, region: str):
        import boto3

        self.table = boto3.resource("dynamodb", region_name=region).Table(table)

    def for_claim(self, claim_id: str) -> list[dict]:
        from boto3.dynamodb.conditions import Key

        return self.table.query(KeyConditionExpression=Key("claim_id").eq(claim_id), ScanIndexForward=True)["Items"]

    def append(self, claim_id: str, event: str, actor: str, payload: dict) -> dict:
        history = self.for_claim(claim_id)
        entry = {
            "claim_id": claim_id, "seq": len(history) + 1, "ts": datetime.now(timezone.utc).isoformat(),
            "event": event, "actor": actor, "payload": json.loads(json.dumps(payload, default=str)),
            "prev_hash": history[-1]["hash"] if history else "0" * 64,
        }
        entry["hash"] = _hash(entry)
        # Conditional put: two writers can't both claim the same sequence number.
        self.table.put_item(Item=entry, ConditionExpression="attribute_not_exists(seq)")
        return entry
