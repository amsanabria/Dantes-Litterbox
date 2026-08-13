import os
import json
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

BOT_TOKEN = os.environ.get("BOT_TOKEN")

MODEL_NAME = "vikhyatk/moondream2"
MODEL_REVISION = "2024-08-26"

_model = None
_tokenizer = None

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
    "Megadrive"
    "Master System"
    "Dreamcast",
    # Other
    "PC",
    "Unknown",
]

def get_model():
    """Carga el modelo una sola vez por ejecución (lazy load)."""
    global _model, _tokenizer
    if _model is None:
        print("Cargando modelo...")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        print("Modelo cargado.")
    return _model, _tokenizer


def download_telegram_photo(file_id: str, dest_path: str = "foto.jpg") -> str:
    resp = requests.get(
        f"https://api.telegram.org/bot{BOT_TOKEN}/getFile",
        params={"file_id": file_id},
        timeout=10,
    )
    resp.raise_for_status()
    file_path = resp.json()["result"]["file_path"]

    file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    file_resp = requests.get(file_url, timeout=10)
    file_resp.raise_for_status()

    with open(dest_path, "wb") as f:
        f.write(file_resp.content)

    return dest_path


def identify_cover_local(image_path: str) -> dict:
    model, tokenizer = get_model()
    image = Image.open(image_path)
    enc_image = model.encode_image(image)

    # 1. Classify movie or videogame
    media_type = model.answer_question(
            enc_image,
            "CLASSIFICATION TASK. "
            "This image is either a video game cover or a movie cover. "
            "If it is a video game, answer GAME. "
            "If it is a movie, answer MOVIE. "
            "Answer with exactly one word: GAME or MOVIE.",
            tokenizer,
        ).strip().upper()

    # 2. Identify title
    title = model.answer_question(
        enc_image,
        "What is the exact title shown on this cover? Don't say anything else, just the title",
        tokenizer,
    ).strip() 
    

    if "GAME" in media_type:
        media_type = "game"
    elif "MOVIE" in media_type:
        media_type = "movie"
    else:
        media_type = "unknown"

    # 3. If videogame identify platform
    platforms = ", ".join(VIDEOGAME_PLATFORMS)

    if media_type == "game":
        platform = model.answer_question(
            enc_image,
            "PLATFORM CLASSIFICATION TASK. "
            "This is a video game cover. "
            "Identify the platform or console shown or indicated by the cover. "
            f"Choose exactly one from: {platforms}. "
            "Answer with exactly one option from the list." \
            "If no platform suits answer 'unknown'",
            tokenizer,
        ).strip()

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

    print("Respuesta título:", title)
    print("Respuesta tipo:", media_type)
    print("Respuesta plataforma:", platform)

    return result

def main():
    file_id = os.environ.get("FILE_ID")

    if not file_id:
        print("No hay foto en este mensaje, nada que identificar.")
        return

    photo_path = download_telegram_photo(file_id)
    result = identify_cover_local(photo_path)

    print("Resultado identificación:", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()