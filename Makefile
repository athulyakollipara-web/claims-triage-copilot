.PHONY: setup pipeline test eval serve docker deploy

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[spark,eval,dev]"

pipeline:   ## generate data -> PySpark features -> train XGBoost -> drift report
	python -m claims_copilot.cli pipeline

test:
	python -m pytest

eval:       ## model vs baselines, explanation consistency, narrative faithfulness, drift -> MLflow
	python eval/run_eval.py

eval-bedrock:
	CLAIMS_LLM=bedrock python eval/run_eval.py

serve:
	uvicorn claims_copilot.api.app:app --reload --port 8000

docker:
	docker build -t claims-triage-copilot . && docker run -p 8000:8000 claims-triage-copilot

deploy:     ## needs AWS credentials, Docker and the CDK CLI (npm i -g aws-cdk)
	cd infra && pip install -e "..[infra]" && cdk deploy
