from openai import OpenAI
from schemas import PrescriptionRecord
from validate import validate_response, Verdict
from collections import Counter

BASE_URL = "http://localhost:5000/v1"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """
You are a precise information-extraction system. Extract the prescription details from the user's text into the required JSON schema. Omit optional
fields that are not present in the text.
"""

def kwargs(system_prompt, user_text):
    args = dict(model = MODEL,
            messages = [{"role":"system", "content":system_prompt},
                        {"role":"user", "content":user_text}],
            response_format={"type":"json_schema", "json_schema":{
                "name" : "prescription",
                "schema": PrescriptionRecord.model_json_schema()}},
            max_tokens=MAX_TOKENS,
            temperature = 0.0
    )
    return args

def gen_and_validate(texts : list[str]):
    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
    counts = Counter()
    for t in texts:
        resp = client.chat.completions.create(**kwargs(SYSTEM_PROMPT, t))
        msg = resp.choices[0]
        counts[validate_response(msg.message.content, msg.finish_reason)] += 1
    return counts
    
if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("paraloq/json_data_extraction", split="train")
    texts = [r["text"] for r in ds if r["topic"] == "medical"]
    counts = gen_and_validate(texts)
    print(counts)
    assert counts[Verdict.VALID] == len(texts), f"NOT 100% valid: {counts}"