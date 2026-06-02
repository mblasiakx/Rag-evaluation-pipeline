import statistics

def chunk_size_mean(chunks: list[str]):  

    if not chunks:
        return 0.0
    return statistics.mean([len(chunk) for chunk in chunks])



def chunk_size_std(chunks: list[str]):
    if not chunks:
        return 0.0
    if len(chunks) < 2:
         return 0.0
    return statistics.stdev([len(chunk) for chunk in chunks])