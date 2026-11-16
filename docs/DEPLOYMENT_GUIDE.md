# Deployment guide: connecting a new insurer's claims data

This is the guide a delivery engineer would hand to a customer's data and security teams. It covers what the copilot needs, what it touches, and the decisions the customer has to make.

## 1. What the copilot needs from you

Six extracts, one per source system. Column names can differ: map them in `claims_copilot/features/build.py` (`_read` calls) rather than renaming upstream.

| System | File | Required columns | Refresh |
|---|---|---|---|
| Policy administration | `policies.csv`, `customers.csv` | policy_id, customer_id, product, issue_date, face_amount, annual_premium; customer dob, name | daily |
| Claims | `claims.csv` | claim_id, policy_id, provider_id, claim_type, loss_date, report_date, claim_amount, cause, channel | hourly or daily |
| Policy servicing | `policy_changes.csv` | policy_id, change_type (beneficiary / coverage_increase / address / bank_account), change_date | daily |
| Provider registry | `providers.csv` | provider_id, provider_type, prior_claims, prior_flagged | weekly |
| Payments | `payments.csv` | claim_id, payee_account_id, payment_method, payee_relationship | daily |
| Document management | `documents.csv`, `adjuster_notes.csv` | claim_id, doc_type, received_date; claim_id, note_text | hourly |

**Labels for training:** closed investigations with an outcome (`is_fraud` = confirmed). Ideally two or more years, with at least a few hundred confirmed cases. With fewer, start with the logistic baseline (it's reported alongside XGBoost) and revisit when more outcomes close.

## 2. Decisions the customer must make

1. **Review capacity.** `REVIEW_CAPACITY` (default 10%) sets how many claims go to enhanced review. Set it from investigator headcount, not model accuracy.
2. **Referral policy.** `claims_copilot/policy.py` maps score + evidence to actions. Claims operations and compliance should sign off on it. The model never pays, denies or refers on its own: every action waits for an investigator.
3. **Required documents per product.** Edit `REQUIRED_DOCS` in `claims_copilot/evidence.py`.
4. **Which model writes summaries.** Any Bedrock model the customer has enabled (`BEDROCK_NARRATIVE_MODEL`, `BEDROCK_FALLBACK_MODEL`). Summaries that cite facts not in the case file are automatically replaced with the template summary.
5. **Data residency.** Choose the AWS region; Bedrock cross-region inference profiles (`us.` model IDs) can route within a geography. Use in-region model IDs if that's not acceptable.

## 3. Security review checklist

- **PII:** adjuster notes pass through Amazon Comprehend `DetectPiiEntities` before any model sees them; the redacted text is what's stored in traces.
- **Identity:** every claim route requires a Cognito JWT. The investigator recorded in the audit log comes from the token, not the request body.
- **Audit:** DynamoDB table with `RETAIN` policy and point-in-time recovery. The Lambda role can only `PutItem` and `Query` on it, so records can't be edited or deleted by the application. Each record carries the previous record's hash.
- **Least privilege:** read-only access to the `data_store/` prefix; Bedrock limited to model invocation; no wildcard DynamoDB access.
- **No training on customer data by the model provider:** Bedrock does not use inputs to train models.

## 4. Rollout plan

| Week | Step | Exit criterion |
|---|---|---|
| 1 | Land extracts in S3, run the feature pipeline as a Glue job, train on history | Hold-out PR-AUC beats the logistic baseline and the current rules |
| 2 | **Shadow mode**: score every new claim, show nothing to investigators | Score distribution and drift stable for 2 weeks |
| 3-4 | Pilot with one investigator team; copilot recommendations visible | Investigator agreement rate and time-to-decision tracked from the audit log |
| 5+ | Expand; monthly retrain review triggered by the PSI alarms | Drift alarms wired to the on-call channel |

## 5. Operating it

- **Drift:** run the PSI report on each scoring batch (`claims_copilot/model/drift.py`) and publish to CloudWatch. The stack creates alarms at PSI > 0.25 for four key features.
- **Retraining:** rerun `cli pipeline` on new closed investigations. Promote only if the new model beats the current one on the most recent quarter.
- **Feedback loop:** `agreed_with_copilot` in the audit log shows where investigators override the copilot. Review overrides monthly; they're the best source of new rules and features.
