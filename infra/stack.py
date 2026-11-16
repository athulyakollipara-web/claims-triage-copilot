"""AWS CDK stack: S3 artifacts bucket, DynamoDB audit + case memory, Lambda API with
Cognito auth, least-privilege IAM, a SageMaker execution role for batch scoring,
and CloudWatch alarms on feature drift."""
from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_authorizers as authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from constructs import Construct

ROOT = Path(__file__).resolve().parents[1]
DRIFT_ALARM_FEATURES = ["claim_to_face", "report_lag_days", "provider_flag_rate", "channel_online"]


class ClaimsCopilotStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(self, "Artifacts", encryption=s3.BucketEncryption.S3_MANAGED,
                           block_public_access=s3.BlockPublicAccess.BLOCK_ALL, enforce_ssl=True, versioned=True,
                           removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        audit = dynamodb.Table(
            self, "Audit",
            partition_key=dynamodb.Attribute(name="claim_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="seq", type=dynamodb.AttributeType.NUMBER),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(point_in_time_recovery_enabled=True),
            deletion_protection=False, removal_policy=RemovalPolicy.RETAIN,  # audit records outlive the stack
        )
        memory = dynamodb.Table(
            self, "CaseMemory",
            partition_key=dynamodb.Attribute(name="claim_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="ts_actor", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST, removal_policy=RemovalPolicy.DESTROY,
        )

        pool = cognito.UserPool(self, "Investigators", self_sign_up_enabled=False,
                                sign_in_aliases=cognito.SignInAliases(email=True), removal_policy=RemovalPolicy.DESTROY)
        client = pool.add_client("ApiClient", auth_flows=cognito.AuthFlow(user_password=True, user_srp=True))

        fn = lambda_.DockerImageFunction(
            self, "CopilotFn",
            code=lambda_.DockerImageCode.from_image_asset(str(ROOT), file="Dockerfile.lambda",
                                                          exclude=["data_store", "mlruns", "mlflow.db", "cdk.out", ".venv", ".git", "eval/results"]),
            memory_size=2048, timeout=Duration.seconds(60), ephemeral_storage_size=None,
            environment={
                "CLAIMS_MODE": "aws", "CLAIMS_LLM": "bedrock", "CLAIMS_DATA_DIR": "/tmp/data_store",
                "CLAIMS_S3_BUCKET": bucket.bucket_name, "CLAIMS_S3_PREFIX": "data_store",
                "AUDIT_TABLE": audit.table_name, "MEMORY_TABLE": memory.table_name,
            },
        )
        bucket.grant_read(fn, "data_store/*")
        audit.grant(fn, "dynamodb:PutItem", "dynamodb:Query")  # append + read only: no update or delete
        memory.grant_read_write_data(fn)
        fn.add_to_role_policy(iam.PolicyStatement(actions=["comprehend:DetectPiiEntities"], resources=["*"]))
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:Converse"],
            resources=["arn:aws:bedrock:*::foundation-model/*", f"arn:aws:bedrock:*:{self.account}:inference-profile/*"],
        ))

        api = apigw.HttpApi(self, "Api", api_name="claims-copilot")
        auth = authorizers.HttpUserPoolAuthorizer("CognitoAuth", pool, user_pool_clients=[client])
        integ = integrations.HttpLambdaIntegration("CopilotIntegration", fn)
        for path, method in [("/queue", apigw.HttpMethod.GET), ("/claims/{claim_id}/triage", apigw.HttpMethod.POST),
                             ("/claims/{claim_id}/decision", apigw.HttpMethod.POST), ("/claims/{claim_id}/audit", apigw.HttpMethod.GET)]:
            api.add_routes(path=path, methods=[method], integration=integ, authorizer=auth)
        api.add_routes(path="/health", methods=[apigw.HttpMethod.GET], integration=integ)

        # Role for SageMaker batch scoring jobs (see deploy/sagemaker_batch.py).
        sm_role = iam.Role(self, "SageMakerBatchRole", assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"))
        bucket.grant_read_write(sm_role)
        sm_role.add_to_policy(iam.PolicyStatement(
            actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "ecr:GetAuthorizationToken",
                     "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "cloudwatch:PutMetricData"],
            resources=["*"],
        ))

        for feature in DRIFT_ALARM_FEATURES:
            cw.Alarm(self, f"Drift-{feature.replace('_', '-')}",
                     metric=cw.Metric(namespace="ClaimsCopilot/Drift", metric_name="PSI", dimensions_map={"Feature": feature},
                                      statistic="Maximum", period=Duration.days(1)),
                     threshold=0.25, evaluation_periods=1, comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
                     treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
                     alarm_description=f"PSI for {feature} above 0.25: review the model for retraining")

        CfnOutput(self, "ApiUrl", value=api.api_endpoint)
        CfnOutput(self, "ArtifactsBucket", value=bucket.bucket_name)
        CfnOutput(self, "UserPoolId", value=pool.user_pool_id)
        CfnOutput(self, "UserPoolClientId", value=client.user_pool_client_id)
        CfnOutput(self, "SageMakerRoleArn", value=sm_role.role_arn)
