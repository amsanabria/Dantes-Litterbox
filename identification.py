import html
import json

from PIL import Image

from models import get_model
from config import MOVIE_FORMATS, VIDEOGAME_PLATFORMS, ICONS


def identify_cover_local(image_path: str) -> dict:

    model, tokenizer = get_model()

    image = Image.open(image_path)

    enc_image = model.encode_image(image)

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

    print(f"Media type: {media_type}")

    title = model.answer_question(
        enc_image,
        "What is the exact title shown on this cover? "
        "Don't say anything else, just the title.",
        tokenizer,
    ).strip()

    print(f"Title: {title}")

    options = "Unknown"

    if media_type == "game":
        options = ", ".join(VIDEOGAME_PLATFORMS)

    elif media_type == "movie":
        options = ", ".join(MOVIE_FORMATS)

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

    print(f"Platform/format: {platform}")

    return {
        "type": media_type,
        "title": title,
        "platform": platform,
    }


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

    return (
        f"{icon} <b>{title}</b>\n\n"
        f"<b>Type:</b> {type_label}\n"
        f"<b>{platform_type}:</b> {platform}\n"
    )