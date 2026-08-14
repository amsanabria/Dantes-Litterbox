import os
import re
import json
import html
import base64
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

CORRECTION_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
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


# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

def format_identification_result(result: dict) -> str:

    media_type = result["type"]
    title_raw = result["title"]
    platform_raw = result["platform"]

    title = html.escape(title_raw)
    platform = html.escape(platform_raw)

    if media_type == "movie":
        icon = "🎬"
        type_label = "Movie"
        platform_type = "Format"
    elif media_type == "game":
        icon = "🎮"
        type_label = "Videogame"
        platform_type = "Platform"
    else:
        icon = "❓"
        type_label = "Desconocido"
        platform_type = "Platform"

    # Machine-readable payload, hidden behind a spoiler so it doesn't
    # clutter the message but survives in reply_to_message.text exactly
    # as-is (no newline collapsing, no label/casing guessing needed).
    data_json = json.dumps(
        {
            "type": media_type,
            "title": title_raw,
            "platform": platform_raw,
        },
        ensure_ascii=False,
    )

    data_block = html.escape(data_json)

    return (
        f"{icon} <b>{title}</b>\n\n"
        f"<b>Type:</b> {type_label}\n"
        f"<b>{platform_type}:</b> {platform}\n"
    )


# ---------------------------------------------------------
# Correction model
# ---------------------------------------------------------

TYPE_LABEL_TO_VALUE = {
    "movie": "movie",
    "videogame": "game",
    "desconocido": "unknown",
}

VALID_TYPES = {"game", "movie", "unknown"}


ICONS = ("🎬", "🎮", "❓")


def _parse_legacy_text(text: str) -> dict:
    """
    Fallback for old messages that don't carry the hidden JSON payload
    (e.g. sent before this format existed), or in case the JSON block
    ever gets mangled in transit. Doesn't rely on newlines being
    preserved and is case-insensitive on labels.
    """

    # Grab whatever sits between "title" and the next label
    # (platform/format), regardless of line breaks or capitalisation.
    title_match = re.search(
        r"title\s*:?\s*(.*?)\s*(?=platform|format|$)",
        text,
        re.IGNORECASE,
    )

    type_match = re.search(
        r"(?:media\s*)?type\s*:?\s*(\w+)",
        text,
        re.IGNORECASE,
    )

    platform_match = re.search(
        r"(?:platform|format)(?:\s*/?\s*format)?\s*:?\s*(.*?)$",
        text,
        re.IGNORECASE,
    )

    title = title_match.group(1).strip() if title_match else "Unknown"

    for icon in ICONS:
        if title.startswith(icon):
            title = title[len(icon):].strip()
            break

    raw_type = type_match.group(1).strip().lower() if type_match else "unknown"

    media_type = TYPE_LABEL_TO_VALUE.get(raw_type, raw_type)
    if media_type not in VALID_TYPES:
        media_type = "unknown"

    platform = platform_match.group(1).strip() if platform_match else "Unknown"

    return {
        "type": media_type,
        "title": title or "Unknown",
        "platform": platform or "Unknown",
    }


def parse_identification_message(text: str) -> dict:
    """
    Parse the bot's own previous message back into a structured dict.
    """

    print("===============")
    print(text)
    print("===============")

    # Parse to dict

    return _parse_legacy_text(text)


def correct_identification(
    previous_result: dict,
    correction_text: str,
) -> dict:

    model, tokenizer = get_correction_model()

    system_prompt = (
        "You are a correction assistant for a movie and video game "
        "identification bot. You are given the current identification "
        "and a correction message from the user. "
        "Identify ONLY the fields the user explicitly wants to change. "
        "Never invent information and never include fields the user "
        "did not mention. "
        "Valid values for 'type' are exactly: game, movie, unknown. "
        "Respond with ONLY a JSON object containing just the changed "
        "fields, no explanation, no markdown, no code fences. "
        "If nothing should change, respond with an empty JSON object: {}\n"
        "Examples of valid responses:\n"
        '{"title": "Devil May Cry"}\n'
        '{"platform": "PS2"}\n'
        '{"type": "game", "platform": "PS2"}\n'
        "{}"
    )

    user_prompt = (
        f"Current identification:\n"
        f"{json.dumps(previous_result, ensure_ascii=False)}\n\n"
        f"User correction:\n{correction_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    raw_answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    print(f"Correction model raw answer: {raw_answer}")

    # -------------------------------------------------
    # Try every {...} block found, keep the first that
    # parses as valid JSON (small models sometimes add
    # example text before/after the real answer).
    # -------------------------------------------------

    candidates = re.findall(r"\{.*?\}", raw_answer, re.DOTALL)

    changes = None

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                changes = parsed
                break
        except json.JSONDecodeError:
            continue

    if changes is None:
        print("Could not parse correction JSON: no valid JSON object found")
        changes = {}

    # -------------------------------------------------
    # Merge: start from the previous (known-good) result
    # and only overwrite fields the model says changed.
    # -------------------------------------------------

    corrected = dict(previous_result)

    if "type" in changes:
        new_type = str(changes["type"]).strip().lower()

        if new_type in VALID_TYPES:
            corrected["type"] = new_type
        else:
            print(
                f"Invalid type '{new_type}' returned by model, "
                "keeping previous type."
            )

    if "title" in changes:
        new_title = str(changes["title"]).strip()

        if new_title:
            corrected["title"] = new_title

    if "platform" in changes:
        new_platform = str(changes["platform"]).strip()

        if new_platform:
            corrected["platform"] = new_platform

    return corrected


def handle_correction(
    chat_id: str,
    message_id: str,
    correction_text: str,
    reply_to_message_text: str,
):

    previous_result = parse_identification_message(
        reply_to_message_text
    )

    corrected_result = correct_identification(
        previous_result=previous_result,
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

def decode_b64_env(var_name: str) -> str:
    """
    Decode a base64-encoded environment variable. The Cloudflare Worker
    base64-encodes any free-text field (message text, reply text)
    before dispatching to GitHub, because passing raw multiline text
    through `${{ }}` into a workflow `env:` mapping is fragile
    (newlines/special chars can get mangled). Returns "" if the var
    is missing or empty.
    """

    raw = os.environ.get(var_name)

    if not raw:
        return ""

    try:
        return base64.b64decode(raw).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        print(f"Could not decode {var_name} as base64: {exc}")
        return ""


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