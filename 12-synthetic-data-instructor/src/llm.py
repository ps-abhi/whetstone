from openai import AsyncOpenAI
import instructor
import asyncio
from schemas import GeneratedExample
from dotenv import load_dotenv
import os
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential
import json

load_dotenv()

openai_client = AsyncOpenAI(base_url = os.getenv("base_url"),
                       api_key = os.getenv("api_key"))

client = instructor.from_openai(openai_client)

limiter = AsyncLimiter(10, 1)

system_prompt = """
You generate ONE realistic example for a tool-calling training dataset. You're given a list of available tools (name, description, typed parameters). 
(1) Invent a single natural user request that exactly ONE of the tools can fully satisfy. 
(2) Produce the correct call: the tool's name and an arguments object whose keys are valid parameters of that tool, with plausible, type-correct values — all required params, optional ones only if the request implies them.
Rules: the query must sound like a real person (vary phrasing/specificity/tone); never mention tool/function/API names in the query; use only parameters that exist on the chosen tool; values must match the declared types.
"""


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate(tools, scenario):
    tools_json = json.dumps(tools, indent=2)
    user_prompt = (
        f"Available tools \n {tools_json} \n\n"
        f"Scenario hint \n {scenario} \n\n"
        "Generate the example"
    )
    async with limiter:
        return await client.create(
            model = os.getenv("model_name"),
            response_model=GeneratedExample,
            temperature = 0.8,
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":user_prompt}
            ]
        )

