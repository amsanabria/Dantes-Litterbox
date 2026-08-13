import os
import json
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

BOT_TOKEN = os.environ.get("BOT_TOKEN")

# ---------------------------------------------------------
# Moondream
# ---------------------------------------------------------

MODEL_NAME = "vikhyatk/moondream2"
MODEL_REVISION = "2024-08-26"

_model = None
_tokenizer = None

# ---------------------------------------------------------
# Qwen correction model
# ---------------------------------------------------------

CORRECTION_MODEL_NAME = "Qwen/Qwen3-0.6B"
CORRECTION_MODEL_REVISION = "main"

_correction_model = None
_correction_tokenizer = None

# ---------------------------------------------------------
# Classification options
# ---------------------------------------------------------

MOVIE_FORMATS = [
    "DVD",
    "Blu-Ray",
    "4K Blu-Ray",
    "Criterion Collection DVD",
    "Criterion Collection Blu-Ray",
    "Criterion Collection 4K Blu-Ray",
]

VIDEOGAME_PLATFORMS = [
    # Playstation
    "PlayStation 1",
    "PlayStation 2",
    "PlayStation 3",
    "PlayStation 4",
    "PlayStation 5",
    "PSP",
    "PS Vita",

    # XBOX
    "Xbox",
    "Xbox 360",
    "Xbox One",
    "Xbox Series X/S",

    # Nintendo
    "NES",
    "SNES",
    "Nintendo 64",
    "GameCube",
    "Wii",
    "Wii U",
    "Switch",
    "Switch 2",
    "Game Boy",
    "Game Boy Color",
    "Game Boy Advance",
    "Nintendo DS",
    "Nintendo 3DS",

    # Sega
    "Megadrive",
    "Master System",
    "Dreamcast",

    # Other
    "PC",
    "Unknown",
]


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

def get_model():
    """Load the vision model once per execution."""
    global _model, _tokenizer

    if _model is None:
        print("Loading vision model...")

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
        )

        print("Vision model loaded.")

    return _model, _tokenizer


def get_correction_model():
    """Load the small language model once per execution."""
    global _correction_model, _correction_tokenizer

    if _correction_model is None:
        print("Loading correction model...")

        _correction_model = AutoModelForCausalLM.from_pretrained(
            CORRECTION_MODEL_NAME,
            revision=CORRECTION_MODEL_REVISION,
        )

        _correction_tokenizer = AutoTokenizer.from_pretrained(
            CORRECTION_MODEL_NAME,
            revision=CORRECTION_MODEL_REVISION,
        )

        print("Correction model loaded.")

    return _correction_model, _correction_tokenizer


# ---------------------------------------------------------
# Telegram
# ---------------------------------------------------------

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

    file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"

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


# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

def format_identification_result(result: dict) -> str:

    return (
        f"Media type: {result['type']}\n"
        f"Title: {result['title']}\n"
        f"Platform/format: {result['platform']}"
    )


# ---------------------------------------------------------
# Correction model
# ---------------------------------------------------------

def correct_identification(
    previous_text: str,
    correction_text: str,
) -> dict:

    model, tokenizer = get_correction_model()

    prompt = f"""You are a correction assistant for a movie and video game identification bot.

The bot's previous identification was:

{previous_text}

The user replied with this correction:

{correction_text}

Update ONLY the fields that the user explicitly corrects.

Keep the previous values for fields that the user does not correct.

Do not invent information.

The valid type values are:
game
movie
unknown

Return ONLY valid JSON in exactly this format:

{{
  "type": "...",
  "title": "...",
  "platform": "..."
}}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    raw_answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    print(
        f"Correction model raw answer: {raw_answer}"
    )

    try:
        start = raw_answer.find("{")
        end = raw_answer.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")

        result = json.loads(
            raw_answer[start:end + 1]
        )

        return {
            "type": str(
                result.get("type", "unknown")
            ).strip().lower(),

            "title": str(
                result.get("title", "Unknown")
            ).strip(),

            "platform": str(
                result.get("platform", "Unknown")
            ).strip(),
        }

    except (json.JSONDecodeError, ValueError) as exc:

        print(
            f"Could not parse correction JSON: {exc}"
        )

        return {
            "type": "unknown",
            "title": "Unknown",
            "platform": "Unknown",
        }


def handle_correction(
    chat_id: str,
    message_id: str,
    correction_text: str,
    reply_to_message_text: str,
):

    corrected_result = correct_identification(
        previous_text=reply_to_message_text,
        correction_text=correction_text,
    )

    response_text = format_identification_result(
        corrected_result
    )

    sent_message = send_telegram_message(
        chat_id=chat_id,
        text=response_text,
        reply_to_message_id=message_id,
        include_buttons=True,
    )

    print(
        "Corrected identification:",
        json.dumps(
            corrected_result,
            ensure_ascii=False,
        ),
    )

    print(
        "Corrected result sent to Telegram. "
        f"Message ID: {sent_message['result']['message_id']}"
    )


# ---------------------------------------------------------
# Moondream identification
# ---------------------------------------------------------

def identify_cover_local(image_path: str) -> dict:

    model, tokenizer = get_model()

    image = Image.open(image_path)

    enc_image = model.encode_image(image)

    # -----------------------------------------------------
    # 1. Classify movie or videogame
    # -----------------------------------------------------

    media_type = model.answer_question(
        enc_image,
        "CLASSIFICATION TASK. "
        "This image is either a video game cover or a movie cover. "
        "If it is a video game, answer GAME. "
        "If it is a movie, answer MOVIE. "
        "Answer with exactly one word: GAME or MOVIE.",
        tokenizer,
    ).strip().upper()

    if "GAME" in media_type:
        media_type = "game"

    elif "MOVIE" in media_type:
        media_type = "movie"

    else:
        media_type = "unknown"

    print(
        f"Media type: {media_type}"
    )

    # -----------------------------------------------------
    # 2. Identify title
    # -----------------------------------------------------

    title = model.answer_question(
        enc_image,
        "What is the exact title shown on this cover? "
        "Don't say anything else, just the title.",
        tokenizer,
    ).strip()

    print(
        f"Title: {title}"
    )

    # -----------------------------------------------------
    # 3. Identify platform / format
    # -----------------------------------------------------

    options = "Unknown"

    if media_type == "game":
        options = ", ".join(
            VIDEOGAME_PLATFORMS
        )

    elif media_type == "movie":
        options = ", ".join(
            MOVIE_FORMATS
        )

    platform = model.answer_question(
        enc_image,
        "PLATFORM/FORMAT CLASSIFICATION TASK. "
        f"This is a {media_type} cover. "
        "Identify the platform or physical format shown "
        "or indicated by the cover. "
        f"Choose exactly one from: {options}. "
        "Answer with exactly one option from the list. "
        "If no option is suitable or no platform/format "
        "is visible, answer 'Unknown'.",
        tokenizer,
    ).strip()

    print(
        f"Platform/format: {platform}"
    )

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    result = {
        "type": media_type,
        "title": title,
        "platform": platform,
        "raw": {
            "title": title,
            "type": media_type,
            "platform": platform,
        },
    }

    return result


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    file_id = os.environ.get("FILE_ID")
    chat_id = os.environ.get("CHAT_ID")
    message_id = os.environ.get("MESSAGE_ID")

    text = os.environ.get(
        "TEXT",
        "",
    )

    reply_to_message_id = os.environ.get(
        "REPLY_TO_MESSAGE_ID"
    )

    reply_to_message_text = os.environ.get(
        "REPLY_TO_MESSAGE_TEXT"
    )

    if not chat_id:
        print("No CHAT_ID.")
        return

    # -----------------------------------------------------
    # User replied to a previous bot identification
    # -----------------------------------------------------

    if (
        reply_to_message_id
        and reply_to_message_text
        and text.strip()
    ):

        print(
            f"Handling correction to message "
            f"{reply_to_message_id}..."
        )

        handle_correction(
            chat_id=chat_id,
            message_id=message_id,
            correction_text=text.strip(),
            reply_to_message_text=reply_to_message_text,
        )

        return

    # -----------------------------------------------------
    # New image
    # -----------------------------------------------------

    if not file_id:
        print(
            "No photo or correction in the message."
        )
        return

    # -----------------------------------------------------
    # Process image
    # -----------------------------------------------------

    photo_path = download_telegram_photo(
        file_id
    )

    result = identify_cover_local(
        photo_path
    )

    print(
        "Identification result:",
        json.dumps(
            result,
            ensure_ascii=False,
        ),
    )

    # -----------------------------------------------------
    # Send result to Telegram
    # -----------------------------------------------------

    response_text = format_identification_result(
        result
    )

    sent_message = send_telegram_message(
        chat_id=chat_id,
        text=response_text,
        reply_to_message_id=message_id,
        include_buttons=True,
    )

    sent_message_id = sent_message[
        "result"
    ][
        "message_id"
    ]

    print(
        "Result sent to Telegram. "
        f"Message ID: {sent_message_id}"
    )


if __name__ == "__main__":
    main()