from jinja2 import Environment, FileSystemLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

def load_prompt(prompt_path: str):
    path = Path(prompt_path).resolve()  # <-- absolutna ścieżka
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    PROMPTS_BASE = (BASE_DIR / "experiments/prompts_experiments/prompts").resolve()  # <-- absolutna
    env = Environment(loader=FileSystemLoader(str(PROMPTS_BASE)))

    # Ścieżka względna względem loadera
    template_path_relative = path.relative_to(PROMPTS_BASE)
    template = env.get_template(str(template_path_relative).replace("\\", "/"))
    return template