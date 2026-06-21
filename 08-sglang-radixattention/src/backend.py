import sglang as sgl
import re 
from pathlib import Path

prompt_file = Path(__file__).parent.parent/"prompts"/"prompt.txt"

with open(prompt_file, encoding="utf-8") as f:
    prompt = f.read()

@sgl.function
def self_consistency(s, prompt, n, max_tokens, temperature):
    s += prompt #shared prefix added before the prompt
    forks = s.fork(n)
    for f in forks:
        f += sgl.gen("answer", max_tokens=max_tokens, temperature=temperature, top_p=0.95)
    s["answers"] = [f["answer"] for f in forks]


if __name__=="__main__":
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    state = self_consistency.run(prompt=prompt, n=8, max_tokens=512, temperature=0.8)
    print(state["answers"])