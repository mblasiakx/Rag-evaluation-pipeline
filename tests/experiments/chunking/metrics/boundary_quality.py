def boundary_quality(chunks: list[str]) -> float:
    if not chunks:
        return 0.0
    count = 0
    for chunk in chunks:
        if not chunk.strip():
            continue
        last_char = chunk.strip()[-1]
        if last_char in {".", "?", "!"}:
            count += 1
    
    return count / len(chunks)