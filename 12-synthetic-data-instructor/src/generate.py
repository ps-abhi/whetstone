import json
import random 
from pathlib import Path
import asyncio
from llm import generate
from schemas import Record


catalog_path = Path(__file__).parent.parent/"seeds"/"tool_catalog.json"
scenarios_path = Path(__file__).parent.parent/"seeds"/"scenarios.json"
OUT_PATH = Path(__file__).parent.parent/"outputs"/"data"/"raw.jsonl"

TARGET = 10
CHUNK = 5

with open(catalog_path, encoding="utf-8") as f:
    catalog = json.load(f)

with open(scenarios_path, encoding="utf-8") as f:
    scenarios = json.load(f)


def count_existing():
    if not OUT_PATH.exists():
        return 0
    else:
        with open(OUT_PATH, encoding="utf-8") as f:
            return sum(1 for line in f) # a neat trick i found where we generate a 1 for each line, sum then adds it all up
        
async def make_one(): #generate 1 record
    sample = random.sample(catalog, k=5)
    scenario = random.choice(scenarios)
    try:
        generate_example = await generate(sample, scenario)
        return Record(available_tools=sample, user_query=generate_example.user_query, tool_calls=generate_example.tool_calls)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return None  # no record gets added


async def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = count_existing()
    print(f"Resuming at {done}/{TARGET}")

    while done<TARGET:
        n = min(CHUNK, TARGET-done)
        results = await asyncio.gather(*(make_one() for i in range(n))) # * is the unpacking operator 
        valid = [r for r in results if r is not None]

        with open(OUT_PATH, "a", encoding="utf-8") as f:
            for r in valid:
                f.write(r.model_dump_json() + "\n")
        done +=len(valid)
        print(f"{len(valid)}/{n} (rejected {n-len(valid)}). Total done {done}/{TARGET}")

if __name__ == "__main__":
    asyncio.run(main())
