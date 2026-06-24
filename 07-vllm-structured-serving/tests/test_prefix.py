from bench import shared_workload, unique_workload, build_prefix, tokenizer

def test_shared_is_identical():
    shared = shared_workload(["a", "b", "c"], 512)
    assert len({r.system for r in shared}) == 1 

def test_unique_is_distinct():
    unique = unique_workload(["a", "b", "c"], 512)
    assert len({r.system for r in unique}) == 3   

def test_length():
    n = len(tokenizer.encode(build_prefix(1024)))
    assert 1000 <= n <= 1100