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
        tools = {t.name : {p.name: p.required for p in t.parameter} for t in self.available_tools}
        for call in self.tool_calls:
            if call.name not in tools:
                raise ValueError(f"{call.name} Tool doesn't exist")
            spec = tools[call.name]
            if not set(call.arguments) <= set(spec):
                raise ValueError("Args returned lesser.")
            
            required = {name for name, req in spec.items() if req}
            missing = required - set(call.arguments)
            if missing:
                raise ValueError(f"{call.name} is missing required args: {missing}")

        return self

class GeneratedExample(BaseModel):
    user_query: str
    tool_calls: list[ToolCall]
