import os


def _env_int(name: str, default: int) -> int:
    raw_value = str(os.getenv(name, default)).strip()
    try:
        return int(raw_value)
    except ValueError:
        return default


DB_CONFIG = {
    "HOST": str(os.getenv("DB_HOST", "localhost")).strip(),
    "USER": str(os.getenv("DB_USER", "root")).strip(),
    "PASSWORD": str(os.getenv("DB_PASSWORD", "")).strip(),
    "DATABASE": str(os.getenv("DB_NAME", "aiconversation")).strip(),
    "PORT": _env_int("DB_PORT", 3306),
}