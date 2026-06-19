from typing import Any

from pydantic import BaseModel, model_validator


class Parameter(BaseModel):
    name : str
    type : str
    required : bool 
    description : str

class ToolSpec(BaseModel):
    name : str
    description : str
    parameter : list[Parameter]

class ToolCall(BaseModel):
    name : str
    arguments : dict[str, Any]


class Record(BaseModel):
    available_tools : list[ToolSpec]
    user_query : str
    tool_calls : list[ToolCall]
    
    @model_validator(mode="after")
    def check_calls(self):
        tools = {t.name : {p.name for p in t.parameter} for t in self.available_tools}
        for call in self.tool_calls:
            if call.name not in tools:
                raise ValueError(f"{call.name} Tool doesn't exist")
                
            if not set(call.arguments) <= tools[call.name]:
                raise ValueError("Args returned lesser.")
        return self

class GeneratedExample(BaseModel):
    user_query: str
    tool_calls: list[ToolCall]
