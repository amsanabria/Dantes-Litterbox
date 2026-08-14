from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODEL_NAME,
    MODEL_REVISION,
    CORRECTION_MODEL_NAME,
    CORRECTION_MODEL_REVISION,
)


_model = None
_tokenizer = None

_correction_model = None
_correction_tokenizer = None


def get_model():
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
