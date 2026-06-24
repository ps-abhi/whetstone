import json
from schemas import PrescriptionRecord
from pydantic import ValidationError
from enum import Enum

class Verdict(str, Enum):
    VALID = "valid"
    TRUNCATED = "truncated"
    PARSE_ERROR = "parse_error"     #not even JSON 
    SCHEMA_ERROR = "schema_error" # parses but wrong shape 

def validate_response(text, finish_reason=None):
    if finish_reason == "length":
        return Verdict.TRUNCATED
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return Verdict.PARSE_ERROR
    try:
        PrescriptionRecord.model_validate(obj)
    except ValidationError:
        return Verdict.SCHEMA_ERROR
    return Verdict.VALID