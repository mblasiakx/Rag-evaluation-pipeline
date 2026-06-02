def faithfulness(pred, context):
    """
    Procent tokenów predykcji, które występują w context
    """
    pred_tokens = set(pred.lower().split())
    context_tokens = set(context.lower().split())
    if not pred_tokens:
        return 0.0
    correct = len(pred_tokens & context_tokens)
    return correct / len(pred_tokens)

def coverage(gold, context):
    """
    Procent tokenów gold answer obecnych w context
    """
    gold_tokens = set(gold.lower().split())
    context_tokens = set(context.lower().split())
    if not gold_tokens:
        return 0.0
    covered = len(gold_tokens & context_tokens)
    return covered / len(gold_tokens)

def hallucination_rate(pred, context):
    """
    Procent tokenów predykcji, które NIE występują w context
    """
    pred_tokens = set(pred.lower().split())
    context_tokens = set(context.lower().split())
    if not pred_tokens:
        return 0.0
    hallucinated = len(pred_tokens - context_tokens)
    return hallucinated / len(pred_tokens)

def token_overlap(pred, context):
    """
    Liczba wspólnych tokenów pred i context
    """
    pred_tokens = set(pred.lower().split())
    context_tokens = set(context.lower().split())
    return len(pred_tokens & context_tokens)

def context_utilization_ratio(pred, context):
    """
    Ile tokenów context zostało użytych w odpowiedzi
    """
    pred_tokens = set(pred.lower().split())
    context_tokens = set(context.lower().split())
    if not context_tokens:
        return 0.0
    used = len(pred_tokens & context_tokens)
    return used / len(context_tokens)