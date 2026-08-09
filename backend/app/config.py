import os

from dotenv import load_dotenv


load_dotenv()


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Required environment variable is missing: {name}"
        )

    return value


def integer_env(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()

    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {name} must be an integer."
        ) from exc


HOST = os.getenv(
    "HOST",
    "0.0.0.0",
).strip()


PORT = integer_env(
    name="PORT",
    default=8000,
)


AZURE_ML_ENDPOINT = required_env(
    "AZURE_ML_ENDPOINT",
)


AZURE_ML_API_KEY = required_env(
    "AZURE_ML_API_KEY",
)


REQUEST_TIMEOUT = integer_env(
    name="REQUEST_TIMEOUT",
    default=30,
)


ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]