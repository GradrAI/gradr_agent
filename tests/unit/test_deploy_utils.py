from app.app_utils.deploy import display_env_value, is_sensitive_env_name


def test_sensitive_env_names_are_masked() -> None:
    assert is_sensitive_env_name("MDB_MCP_CONNECTION_STRING") is True
    assert is_sensitive_env_name("GOOGLEAI_API_KEY") is True
    assert is_sensitive_env_name("JWT_SECRET_KEY") is True
    assert is_sensitive_env_name("ADMIN_JWT_TOKEN") is True

    assert display_env_value("MDB_MCP_CONNECTION_STRING", "mongodb+srv://user:pass@example/db") == "***"


def test_non_sensitive_env_names_are_printed() -> None:
    assert is_sensitive_env_name("GOOGLE_CLOUD_LOCATION") is False
    assert is_sensitive_env_name("GRADR_LIGHTWEIGHT_GEMINI_MODEL") is False

    assert display_env_value("GOOGLE_CLOUD_LOCATION", "global") == "global"
    assert display_env_value("GRADR_LIGHTWEIGHT_GEMINI_MODEL", "gemini-3.1-flash-lite") == "gemini-3.1-flash-lite"
