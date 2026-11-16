#!/usr/bin/env python3
import aws_cdk as cdk

from stack import ClaimsCopilotStack

app = cdk.App()
ClaimsCopilotStack(app, "ClaimsTriageCopilot")
app.synth()
