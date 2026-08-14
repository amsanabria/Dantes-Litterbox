import os
import json

from telegram import (
    download_telegram_photo,
    send_telegram_message,
)

from identification import (
    identify_cover_local,
    format_identification_result,
)

from correction import handle_correction
from utils import decode_b64_env


def main():

    file_id = os.environ.get("FILE_ID")
    chat_id = os.environ.get("CHAT_ID")
    message_id = os.environ.get("MESSAGE_ID")

    text = decode_b64_env("TEXT_B64")

    reply_to_message_id = os.environ.get(
        "REPLY_TO_MESSAGE_ID"
    )

    reply_to_message_text = decode_b64_env(
        "REPLY_TO_MESSAGE_TEXT_B64"
    ) or None

    if not chat_id:
        print("No CHAT_ID.")
        return

    # Correction
    if (
        reply_to_message_id
        and reply_to_message_text
        and text.strip()
    ):
        handle_correction(
            chat_id=chat_id,
            message_id=message_id,
            correction_text=text.strip(),
            reply_to_message_text=reply_to_message_text,
        )
        return

    # New image
    if not file_id:
        print("No photo or correction in the message.")
        return

    photo_path = download_telegram_photo(file_id)

    result = identify_cover_local(photo_path)

    print(
        "Identification result:",
        json.dumps(result, ensure_ascii=False),
    )

    response_text = format_identification_result(result)

    sent_message = send_telegram_message(
        chat_id=chat_id,
        text=response_text,
        reply_to_message_id=message_id,
        include_buttons=True,
    )

    sent_message_id = sent_message["result"]["message_id"]

    print(
        f"Result sent to Telegram. Message ID: {sent_message_id}"
    )


if __name__ == "__main__":
    main()