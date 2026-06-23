import pytest
from inference import loader, MODEL

@pytest.fixture(scope="session")
def model_and_tokenizer():
    return loader(MODEL, dtype="bfloat16")
