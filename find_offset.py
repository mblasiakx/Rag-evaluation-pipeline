from pathlib import Path
import re

TEXT_PATH = "data/metrics_test1.txt"

# wczytaj cały tekst
text = Path(TEXT_PATH).read_text(encoding="utf-8")

# lista fragmentów, które chcemy znaleźć
fragments = [
    "Rome was founded around 625 BC in the areas of ancient Italy known as Etruria and Latium",
    "The UEFA Champions League came about as a new version of the European Cup and was played for the first time in the 1992-1993 season.",
    "Rome entered its Republican Period in 510 BC",
    "The Champions League trophy can be kept permanently by a club that win the tournament five times or three times in a row (since the rule changed in 2008 it is only a replica)",
    "UEFA Champions League came about as a new version of the European Cup and was played for the first time in the 1992-1993 season",
    "French club Lyon have been demoted to Ligue 2 because of the poor state of their finances.",
    "The club were provisionally demoted by the DNGC, the body which oversees the accounts of French professional football clubs, in November.",
    "Lyon officials including owner John Textor, met with the DNGC on Tuesday but failed to convince the body that the club had sufficiently improved their financial situation to lift the punishment.",
    "Lyon's relegation could prove significant to Crystal Palace's hopes of playing in the Europa League next season."
]

# wyszukiwanie każdego fragmentu w tekście
for frag in fragments:
    matches = list(re.finditer(re.escape(frag), text))
    if not matches:
        print(f"Fragment not found: {frag}")
    else:
        for m in matches:
            print({
                "fragment": frag,
                "start_char": m.start(),
                "end_char": m.end()
            })