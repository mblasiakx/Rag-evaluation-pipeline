from jinja2 import Template

def render_prompt(template, query: str, context: str) -> str:
    return template.render(query=query, context=context)