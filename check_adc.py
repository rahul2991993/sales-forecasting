import google.auth
from google.auth.transport.requests import Request

credentials, project_id = google.auth.default()

credentials.refresh(Request())

print("ADC project:", project_id)
print("Credential type:", type(credentials).__name__)
print(
    "Service account:",
    getattr(credentials, "service_account_email", None)
)
print(
    "Quota project:",
    getattr(credentials, "quota_project_id", None)
)