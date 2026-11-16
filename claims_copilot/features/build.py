"""PySpark feature pipeline: six upstream systems -> one model-ready row per claim.

Runs locally and unchanged as an AWS Glue job (paths become ``s3://``). Every
feature is computed only from information available on the claim's report date,
so there is no leakage from the future (e.g. documents received later).
"""
from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

CATEGORICALS = {
    "claim_type": ["term_life", "whole_life", "disability", "long_term_care"],
    "cause": ["natural", "accident", "illness", "unknown"],
    "channel": ["agent", "online", "phone"],
    "provider_type": ["physician", "funeral_home", "care_facility"],
}
NUMERIC_FEATURES = [
    "policy_age_days", "contestable", "loss_before_issue", "insured_age", "face_amount", "annual_premium",
    "premium_to_face", "claim_amount", "claim_to_face", "report_lag_days", "late_report",
    "beneficiary_change_90d", "coverage_increase_180d", "address_change_90d", "bank_change_60d", "policy_changes_365d",
    "provider_prior_claims", "provider_flag_rate", "payee_third_party", "payment_by_check", "payee_account_claims",
    "docs_required", "docs_received_at_report", "docs_missing_ratio", "customer_prior_claims",
]
FEATURES = NUMERIC_FEATURES + [f"{col}_{v}" for col, values in CATEGORICALS.items() for v in values]


def get_spark() -> SparkSession:
    return (SparkSession.builder.master("local[2]").appName("claims-features")
            .config("spark.ui.enabled", "false").config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.session.timeZone", "UTC").getOrCreate())


def _read(spark: SparkSession, raw: Path, rel: str) -> DataFrame:
    return spark.read.option("header", True).option("inferSchema", True).csv(str(raw / rel))


def _days_before_loss(changes: DataFrame, kind: str, window: int, name: str) -> DataFrame:
    return (changes.where(F.col("change_type") == kind)
            .where((F.col("days_before_loss") > 0) & (F.col("days_before_loss") <= window))
            .groupBy("claim_id").agg(F.lit(1).alias(name)))


def build_features(spark: SparkSession, raw: Path) -> DataFrame:
    claims = _read(spark, raw, "claims/claims.csv").withColumn("loss_date", F.to_date("loss_date")).withColumn("report_date", F.to_date("report_date"))
    policies = _read(spark, raw, "policy_admin/policies.csv").withColumn("issue_date", F.to_date("issue_date"))
    customers = _read(spark, raw, "policy_admin/customers.csv").select("customer_id", F.to_date("dob").alias("dob"))
    changes = _read(spark, raw, "servicing/policy_changes.csv").withColumn("change_date", F.to_date("change_date"))
    providers = _read(spark, raw, "provider_registry/providers.csv")
    payments = _read(spark, raw, "payments/payments.csv")
    docs = _read(spark, raw, "documents/documents.csv").withColumn("received_date", F.to_date("received_date"))

    base = (claims.join(policies.drop("state"), "policy_id").join(customers, "customer_id")
            .join(providers.select("provider_id", "provider_type", "prior_claims", "prior_flagged"), "provider_id", "left")
            .join(payments, "claim_id", "left"))

    claim_changes = (changes.join(claims.select("claim_id", "policy_id", "loss_date"), "policy_id")
                     .withColumn("days_before_loss", F.datediff("loss_date", "change_date")))
    flags = [
        _days_before_loss(claim_changes, "beneficiary", 90, "beneficiary_change_90d"),
        _days_before_loss(claim_changes, "coverage_increase", 180, "coverage_increase_180d"),
        _days_before_loss(claim_changes, "address", 90, "address_change_90d"),
        _days_before_loss(claim_changes, "bank_account", 60, "bank_change_60d"),
        claim_changes.where((F.col("days_before_loss") > 0) & (F.col("days_before_loss") <= 365))
        .groupBy("claim_id").agg(F.count("*").alias("policy_changes_365d")),
    ]
    for f in flags:
        base = base.join(f, "claim_id", "left")

    doc_stats = (docs.join(claims.select("claim_id", "report_date"), "claim_id")
                 .groupBy("claim_id").agg(
                     F.count("*").alias("docs_required"),
                     F.sum(F.when(F.col("received_date") <= F.date_add("report_date", 14), 1).otherwise(0)).alias("docs_received_at_report")))
    acct = payments.groupBy("payee_account_id").agg(F.countDistinct("claim_id").alias("payee_account_claims"))
    prior = (claims.alias("a").join(policies.select("policy_id", "customer_id"), "policy_id")
             .alias("x").join(claims.join(policies.select("policy_id", "customer_id"), "policy_id").alias("y"),
                              (F.col("x.customer_id") == F.col("y.customer_id")) & (F.col("y.report_date") < F.col("x.report_date")), "left")
             .groupBy(F.col("x.claim_id").alias("claim_id")).agg(F.count(F.col("y.claim_id")).alias("customer_prior_claims")))

    df = base.join(doc_stats, "claim_id", "left").join(acct, "payee_account_id", "left").join(prior, "claim_id", "left")

    df = (df
          .withColumn("policy_age_days", F.datediff("loss_date", "issue_date"))
          .withColumn("contestable", (F.col("policy_age_days") < 730).cast("int"))
          .withColumn("loss_before_issue", (F.col("policy_age_days") < 0).cast("int"))
          .withColumn("insured_age", F.floor(F.datediff("loss_date", "dob") / 365.25))
          .withColumn("premium_to_face", F.col("annual_premium") / F.col("face_amount"))
          .withColumn("claim_to_face", F.col("claim_amount") / F.col("face_amount"))
          .withColumn("report_lag_days", F.datediff("report_date", "loss_date"))
          .withColumn("late_report", (F.col("report_lag_days") > 90).cast("int"))
          .withColumn("provider_prior_claims", F.coalesce("prior_claims", F.lit(0)))
          .withColumn("provider_flag_rate", F.coalesce(F.col("prior_flagged") / F.greatest(F.col("prior_claims"), F.lit(1)), F.lit(0.0)))
          .withColumn("payee_third_party", (F.col("payee_relationship") == "third_party").cast("int"))
          .withColumn("payment_by_check", (F.col("payment_method") == "check").cast("int"))
          .withColumn("docs_missing_ratio", 1 - F.col("docs_received_at_report") / F.col("docs_required")))
    for col in ["beneficiary_change_90d", "coverage_increase_180d", "address_change_90d", "bank_change_60d", "policy_changes_365d"]:
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
    for col, values in CATEGORICALS.items():
        for v in values:
            df = df.withColumn(f"{col}_{v}", (F.col(col) == v).cast("int"))

    keep = ["claim_id", "policy_id", "report_date", "claim_type", "claim_amount_raw", "is_fraud"]
    df = df.withColumn("claim_amount_raw", F.col("claim_amount"))
    return df.select(*keep, *[F.col(c).cast("double").alias(c) for c in FEATURES])


def run(raw: Path, out: Path) -> dict:
    spark = get_spark()
    features = build_features(spark, raw)
    features.write.mode("overwrite").parquet(str(out))
    n = features.count()
    return {"rows": n, "features": len(FEATURES), "fraud_rate": round(features.agg(F.avg("is_fraud")).first()[0], 4)}


if __name__ == "__main__":
    from claims_copilot.config import get_settings

    s = get_settings()
    print("history:", run(s.data_dir / "raw", s.data_dir / "features" / "history"))
    print("current:", run(s.data_dir / "raw_current", s.data_dir / "features" / "current"))
