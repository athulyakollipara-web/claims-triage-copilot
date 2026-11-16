"""Case memory: what investigators have said and decided about a claim, across sessions.

Scoped by claim and by investigator, so the copilot can say "you asked for the
physician statement on Tuesday" without leaking one investigator's notes into
another's view unless they're shared. JSON file locally, DynamoDB in AWS mode.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class FileCaseMemory:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def remember(self, claim_id: str, actor: str, note: str, shared: bool = True) -> None:
        with self._lock:
            data = self._load()
            data.setdefault(claim_id, []).append({
                "ts": datetime.now(timezone.utc).isoformat(), "actor": actor, "note": note, "shared": shared,
            })
            self.path.write_text(json.dumps(data, indent=1))

    def recall(self, claim_id: str, actor: str) -> list[dict]:
        return [m for m in self._load().get(claim_id, []) if m["shared"] or m["actor"] == actor]


class DynamoCaseMemory:
    def __init__(self, table: str, region: str):
        import boto3

        self.table = boto3.resource("dynamodb", region_name=region).Table(table)

    def remember(self, claim_id: str, actor: str, note: str, shared: bool = True) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self.table.put_item(Item={"claim_id": claim_id, "ts_actor": f"{ts}#{actor}", "actor": actor, "note": note, "shared": shared, "ts": ts})

    def recall(self, claim_id: str, actor: str) -> list[dict]:
        from boto3.dynamodb.conditions import Key

        items = self.table.query(KeyConditionExpression=Key("claim_id").eq(claim_id))["Items"]
        return [{"ts": i["ts"], "actor": i["actor"], "note": i["note"], "shared": i["shared"]}
                for i in items if i["shared"] or i["actor"] == actor]
