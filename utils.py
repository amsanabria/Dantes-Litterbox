import os
import base64


def decode_b64_env(var_name: str) -> str:

    raw = os.environ.get(var_name)

    if not raw:
        return ""

    try:
        return base64.b64decode(raw).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        print(
            f"Could not decode {var_name} as base64: {exc}"
        )
        return ""
