import json
import shutil
import sys
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parents[1] / "infra"


@pytest.fixture(scope="module")
def template():
    pytest.importorskip("aws_cdk")
    if not shutil.which("node"):
        pytest.skip("CDK synth needs Node.js")
    sys.path.insert(0, str(INFRA))
    import aws_cdk as cdk
    from aws_cdk.assertions import Template
    from stack import ClaimsCopilotStack

    return Template.from_stack(ClaimsCopilotStack(cdk.App(), "Test"))


def test_resources(template):
    template.resource_count_is("AWS::DynamoDB::Table", 2)
    template.resource_count_is("AWS::CloudWatch::Alarm", 4)
    template.resource_count_is("AWS::Cognito::UserPool", 1)


def test_audit_table_is_retained_and_append_only(template):
    tables = template.find_resources("AWS::DynamoDB::Table")
    audit = [t for t in tables.values() if any(k["AttributeName"] == "seq" for k in t["Properties"]["KeySchema"])][0]
    assert audit["DeletionPolicy"] == "Retain"
    audit_id = [k for k, t in tables.items() if t is audit][0]
    audit_actions = set()
    for pol in template.find_resources("AWS::IAM::Policy").values():
        for st in pol["Properties"]["PolicyDocument"]["Statement"]:
            if audit_id in json.dumps(st.get("Resource")):
                acts = st["Action"]
                audit_actions |= set([acts] if isinstance(acts, str) else acts)
    assert audit_actions == {"dynamodb:PutItem", "dynamodb:Query"}  # the copilot can never edit or delete audit records


def test_all_claim_routes_need_auth(template):
    routes = template.find_resources("AWS::ApiGatewayV2::Route")
    for r in routes.values():
        key = r["Properties"]["RouteKey"]
        expected = "NONE" if key == "GET /health" else "JWT"
        assert r["Properties"]["AuthorizationType"] == expected, key


def test_lambda_handler_routes(service):
    import claims_copilot.service as svc_module
    from claims_copilot import lambda_handler

    svc_module.default_service.cache_clear()
    cid = service.store.queue(limit=12)[11]["claim_id"]
    event = {"routeKey": "POST /claims/{claim_id}/triage", "pathParameters": {"claim_id": cid},
             "requestContext": {"authorizer": {"jwt": {"claims": {"email": "jlee@insurer.example"}}}}}
    r = lambda_handler.handler(event, None)
    assert r["statusCode"] == 200
    event["routeKey"] = "POST /claims/{claim_id}/decision"
    event["body"] = json.dumps({"action": "enhanced_review"})
    d = json.loads(lambda_handler.handler(event, None)["body"])
    assert d["status"] == "decided"
    audit = json.loads(lambda_handler.handler({"routeKey": "GET /claims/{claim_id}/audit", "pathParameters": {"claim_id": cid}}, None)["body"])
    assert audit[-1]["actor"] == "jlee@insurer.example"  # identity comes from the JWT, not the body
