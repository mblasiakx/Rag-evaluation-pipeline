from typing import List, Dict

def instruction_features_compliance(response: str, instruction_checks: List[Dict]) -> float:
    """
    Oblicza compliance odpowiedzi modelu względem konkretnych cech instrukcji.
    
    Parametry:
    - response: str – odpowiedź modelu
    - instruction_checks: lista słowników definiujących wymagane cechy instrukcji
        Przykładowy słownik:
        {
            "type": "deny_knowledge" | "single_sentence" | "bullet_points" | "max_words",
            "value": opcjonalnie liczba dla max_words
        }

    Zwraca:
    - float – stosunek spełnionych cech (0.0 - 1.0)
    """
    response_lower = response.lower()
    satisfied = 0

    for check in instruction_checks:
        check_type = check.get("type")

        if check_type == "deny_knowledge":
            # sprawdza, czy model używa zwrotów typu "I don't know"
            if any(kw in response_lower for kw in ["i don't know", "unknown", "cannot answer"]):
                satisfied += 1

        elif check_type == "single_sentence":
            sentences = [s for s in response.strip().split(".") if s.strip()]
            if len(sentences) == 1:
                satisfied += 1

        elif check_type == "bullet_points":
            # sprawdza obecność nowej linii lub znaków punktowych
            if "\n" in response or "-" in response or "•" in response:
                satisfied += 1

        elif check_type == "max_words":
            max_words = check.get("value", 50)  # domyślnie 50 słów
            if len(response.split()) <= max_words:
                satisfied += 1

        else:
            # nieznany typ → ignoruj
            pass

    return satisfied / len(instruction_checks) if instruction_checks else 0.0