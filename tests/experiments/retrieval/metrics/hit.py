def is_hit(query_offsets: list, chunk: dict) -> bool:
    for qo in query_offsets:
        if not (chunk["end_char"] <= qo["start_char"] or chunk["start_char"] >= qo["end_char"]):
            return True
    return False
