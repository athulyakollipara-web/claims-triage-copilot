"""Daily batch scoring with Amazon SageMaker Batch Transform (AWS mode).

Packages ``model.json`` in the layout SageMaker's built-in XGBoost container expects,
uploads it, registers a SageMaker model and runs a transform job over the day's
feature file (CSV, no header, columns in ``features.json`` order).

NOTE: this script was written against the boto3 SageMaker API but has not been run
in CI (it needs an AWS account). Check the XGBoost container image URI for your
region in the SageMaker docs and confirm the container version can load an
XGBoost JSON model before relying on it.

    python deploy/sagemaker_batch.py --bucket <ArtifactsBucket> --role <SageMakerRoleArn> \
        --image <xgboost-container-image-uri> --features s3://<bucket>/scoring/2025-09-30.csv
"""
from __future__ import annotations

import argparse
import tarfile
import tempfile
import time
from pathlib import Path

import boto3

from claims_copilot.config import get_settings


def package_model(artifacts: Path, out: Path) -> Path:
    tar_path = out / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(artifacts / "model.json", arcname="xgboost-model")
    return tar_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--features", required=True, help="s3:// URI of headerless feature CSV")
    p.add_argument("--instance", default="ml.m5.large")
    args = p.parse_args()

    s = get_settings()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    s3 = boto3.client("s3")
    sm = boto3.client("sagemaker")
    with tempfile.TemporaryDirectory() as tmp:
        tar = package_model(s.artifacts_dir, Path(tmp))
        key = f"models/{stamp}/model.tar.gz"
        s3.upload_file(str(tar), args.bucket, key)

    model_name = f"claims-fraud-risk-{stamp}"
    sm.create_model(ModelName=model_name, ExecutionRoleArn=args.role,
                    PrimaryContainer={"Image": args.image, "ModelDataUrl": f"s3://{args.bucket}/{key}"})
    job = f"claims-scoring-{stamp}"
    sm.create_transform_job(
        TransformJobName=job, ModelName=model_name,
        TransformInput={"DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": args.features}},
                        "ContentType": "text/csv", "SplitType": "Line"},
        TransformOutput={"S3OutputPath": f"s3://{args.bucket}/scores/{stamp}/", "AssembleWith": "Line"},
        TransformResources={"InstanceType": args.instance, "InstanceCount": 1},
    )
    print(f"started {job}; scores will land in s3://{args.bucket}/scores/{stamp}/")


if __name__ == "__main__":
    main()
