import json
import random 
from pathlib import Path
import asyncio
from llm import generate
from schemas import Record


catalog_path = Path(__file__).parent.parent/"seeds"/"tool_catalog.json"
scenarios_path = Path(__file__).parent.parent/"seeds"/"scenarios.json"


with open(catalog_path) as f:
    catalog = json.load(f)

with open(scenarios_path) as f:
    scenarios = json.load(f)

async def main():
    sample = random.sample(catalog,k=5)
    scenario = random.choice(scenarios)
    generated_example = await generate(sample, scenario)
    record = Record(available_tools=sample, user_query=generated_example.user_query, tool_calls=generated_example.tool_calls)
    print(record)

if __name__ == "__main__":
    asyncio.run(main())
