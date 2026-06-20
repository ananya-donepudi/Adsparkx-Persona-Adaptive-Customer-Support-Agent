# API Troubleshooting

## 401 Unauthorized

If a request returns `401 Unauthorized`, verify that the API key is present in the `Authorization` header, the key has not expired, and the environment points to the correct workspace. Regenerate the key if it was copied from an old sandbox.

## Webhook Setup

Webhook endpoints must accept HTTPS POST requests and return a 2xx response within 5 seconds. Validate the signing secret before processing the payload. If delivery retries continue for more than 30 minutes, collect the request ID and escalate to support engineering.

## Rate Limits

Standard accounts receive 600 requests per minute. If responses include `429 Too Many Requests`, back off exponentially and inspect the `Retry-After` header.
