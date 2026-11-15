"""Application version number - declared in ONE place only.

The "Info" tab reads this version via /api/app/update; `installer.iss` (AppVersion) must
match, and tests/test_app_update.py verifies it. When releasing a new version: edit both and then create a tag.
"""
APP_VERSION = "1.0.6"
