import os


BOT_TOKEN = os.environ.get("BOT_TOKEN")

# Vision model
MODEL_NAME = "vikhyatk/moondream2"
MODEL_REVISION = "2024-08-26"

# Correction model
CORRECTION_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CORRECTION_MODEL_REVISION = "main"


MOVIE_FORMATS = [
    "DVD",
    "Blu-Ray",
    "4K Blu-Ray",
    "Criterion Collection DVD",
    "Criterion Collection Blu-Ray",
    "Criterion Collection 4K Blu-Ray",
]


VIDEOGAME_PLATFORMS = [
    "PlayStation 1",
    "PlayStation 2",
    "PlayStation 3",
    "PlayStation 4",
    "PlayStation 5",
    "PSP",
    "PS Vita",

    "Xbox",
    "Xbox 360",
    "Xbox One",
    "Xbox Series X/S",

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

    "Megadrive",
    "Master System",
    "Dreamcast",

    "PC",
    "Unknown",
]


VALID_TYPES = {
    "game",
    "movie",
    "unknown",
}

TYPE_LABEL_TO_VALUE = {
    "movie": "movie",
    "videogame": "game",
    "desconocido": "unknown",
}

ICONS = ("🎬", "🎮", "❓")

JSON_FIELDS = {
    "type",
    "platform",
    "title"
}
