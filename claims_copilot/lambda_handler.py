"""AWS Lambda entry point (API Gateway HTTP API, payload v2).

On cold start the store syncs model artifacts and claim files from S3
(``CLAIMS_S3_BUCKET``) into /tmp. Decisions are recorded from the DynamoDB audit
log, so no in-memory state is needed between requests.
"""
from __future__ import annotations

import json

from claims_copilot.service import ACTIONS, default_service


def _resp(status: int, body) -> dict:
    return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": json.dumps(body, default=str)}


def _investigator(event: dict, body: dict) -> str:
    """Prefer the authenticated Cognito identity over anything in the request body."""
    claims = event.get("requestContext", {}).get("authorizer", {}).get("jwt", {}).get("claims", {})
    return claims.get("email") or claims.get("cognito:username") or body.get("investigator", "unknown")


def handler(event: dict, context) -> dict:
    route = event.get("routeKey", "")
    params = event.get("pathParameters") or {}
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _resp(400, {"error": "Body must be JSON"})
    svc = default_service()

    if route == "GET /health":
        return _resp(200, {"status": "ok", "mode": svc.settings.mode})
    if route == "GET /queue":
        return _resp(200, svc.store.queue(limit=20))
    if route == "POST /claims/{claim_id}/triage":
        try:
            out = svc.triage(params["claim_id"], _investigator(event, body))
        except KeyError:
            return _resp(404, {"error": "unknown claim"})
        out.pop("trace", None)
        return _resp(200, out)
    if route == "POST /claims/{claim_id}/decision":
        if body.get("action") not in ACTIONS:
            return _resp(400, {"error": f"action must be one of {sorted(ACTIONS)}"})
        try:
            out = svc.decide(params["claim_id"], _investigator(event, body), body["action"],
                             body.get("reason", ""), body.get("note", ""), body.get("share_note", True))
        except LookupError as e:
            return _resp(409, {"error": str(e)})
        return _resp(200, out)
    if route == "GET /claims/{claim_id}/audit":
        return _resp(200, svc.audit.for_claim(params["claim_id"]))
    return _resp(404, {"error": f"unknown route {route}"})
