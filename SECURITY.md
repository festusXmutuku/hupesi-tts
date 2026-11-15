# Security Policy

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public issues.

Instead, report them privately via GitHub: go to the repository's **Security** tab and choose **Report a vulnerability**. Only the project owner can read these reports.

Please include:
- The app version and operating system.
- Step-by-step reproduction instructions.
- Impact: what an attacker could do.

You will receive a response within 7 days. When a fix is released, you will be credited if you wish.

## Scope

The most important areas are those that accept connections from outside:
- **Remote access and API key** (`apps/remote_access.py`): authentication bypass, API key exposure.
- **Local API:** whether a malicious website opened on the same machine can read or modify app data (CORS, DNS rebinding).
- **Parsing user-uploaded files:** PDF, DOCX, SRT, audio.
- **Installer and on-demand add-on downloads** (GPU, Kokoro).

## Supported Versions

Only the latest release receives security patches.