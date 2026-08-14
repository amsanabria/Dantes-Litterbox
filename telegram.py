import requests

from config import BOT_TOKEN


def download_telegram_photo(
    file_id: str,
    dest_path: str = "foto.jpg",
) -> str:

    resp = requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/getFile",
        params={"file_id": file_id},
        timeout=10,
    )

    resp.raise_for_status()

    file_path = resp.json()["result"]["file_path"]

    file_url = (
        f"https://api.telegram.org/file/"
        f"bot{BOT_TOKEN}/{file_path}"
    )

    file_resp = requests.get(
        file_url,
        timeout=10,
    )

    file_resp.raise_for_status()

    with open(dest_path, "wb") as f:
        f.write(file_resp.content)

    return dest_path


def send_telegram_message(
    chat_id: str,
    text: str,
    reply_to_message_id: str | None = None,
    include_buttons: bool = False,
) -> dict:

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }

    if reply_to_message_id:
        payload["reply_parameters"] = {
            "message_id": int(reply_to_message_id),
        }

    if include_buttons:
        payload["reply_markup"] = {
            "inline_keyboard": [
                [
                    {
                        "text": "➕ Añadir a Excel",
                        "callback_data": "add_to_excel",
                    }
                ]
            ]
        }

    resp = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json=payload,
        timeout=10,
    )

    resp.raise_for_status()

    return resp.json()
