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

    prompt = (
        "This is the cover of a movie or a video game. "
        "Identify whether it is a movie or a game, and give its exact title. "
        "Answer in the format: TYPE: <movie or game> | TITLE: <exact title>"
    )
    raw_answer = model.answer_question(enc_image, prompt, tokenizer)
    print("Respuesta cruda del modelo:", raw_answer)

    # Parseo simple del formato pedido; si el modelo no lo respeta,
    # devolvemos todo como título y dejamos "unknown" en type.
    result = {"type": "unknown", "title": raw_answer.strip(), "raw": raw_answer}
    try:
        parts = raw_answer.split("|")
        type_part = parts[0].split(":")[1].strip().lower()
        title_part = parts[1].split(":")[1].strip()
        result["type"] = "movie" if "movie" in type_part else "game"
        result["title"] = title_part
    except (IndexError, ValueError):
        pass

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