from typing import List

from typing import List, Dict

def instruction_compliance_rate(
    responses: List[str],
    instruction_checks: List[List[Dict]]
) -> float:
    """
    Calculates Instruction Compliance Rate (ICR).

    responses: list of model outputs
    instruction_checks: list of instruction feature sets per response

    Returns: compliance rate (0.0 - 1.0)
    """

    if len(responses) != len(instruction_checks):
        raise ValueError("Responses and instruction checks must have the same length.")

    compliant = 0

    for response, checks in zip(responses, instruction_checks):
        response_lower = response.lower()
        satisfied = True

        for check in checks:
            check_type = check.get("type")

            if check_type == "single_sentence":
                sentences = [s for s in response.split(".") if s.strip()]
                if len(sentences) != 1:
                    satisfied = False

            elif check_type == "deny_knowledge":
                if not any(kw in response_lower for kw in ["i don't know", "unknown", "cannot answer"]):
                    satisfied = False

            elif check_type == "max_words":
                if len(response.split()) > check.get("value", 50):
                    satisfied = False

        if satisfied:
            compliant += 1

    return compliant / len(responses) if responses else 0.0