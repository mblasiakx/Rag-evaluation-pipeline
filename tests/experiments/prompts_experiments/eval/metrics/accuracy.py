"""

C) Evidence coverage / retrieval fidelity
Czy model wskazał fragmenty tekstu, które odpowiadają na pytanie.
Dotyczy promptów structured output, gdzie masz pole evidence.
Można liczyć np. procent słów w answer obecnych w evidence.

D) Completeness / recall
Czy model odpowiedział na wszystkie części pytania, jeśli pytanie jest złożone.
Typowe przy self-ask lub promptach wymagających kilku kroków.

E) Structured output validation
Sprawdzenie czy odpowiedź pasuje do wymaganego schematu JSON / pola.
Czy JSON parsuje się poprawnie, wszystkie pola są obecne, typ danych jest poprawny.

F) Stability / consistency
Jeśli uruchamiasz prompt kilka razy na tym samym kontekście, czy wynik się powtarza.
Przydaje się do porównania wariantów promptów (variants vs base
"""

def exact_match(pred, gold):
    return pred.strip().lower() == gold.strip().lower()

def precision(pred, gold):
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens:
        return 0.0
    correct = len(pred_tokens & gold_tokens)
    return correct / len(pred_tokens)

def recall(pred, gold):
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return 0.0
    correct = len(pred_tokens & gold_tokens)
    return correct / len(gold_tokens)


def f1_score(pred, gold):
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return f1

def batch_accuracy(preds, golds):
    correct = sum(exact_match(p, g) for p, g in zip(preds, golds))
    return correct / len(golds) if golds else 0.0