import re

REFERENCE = re.compile(r"\[([EЕ]\d+\b[^\[\]\n]{0,120})\]|\(([EЕ]\d+\b[^()\n]{0,120})\)", re.I)
IDENTIFIER = re.compile(r"\b[EЕ](\d{1,4})\b", re.I)
EXPLICIT_POINT = re.compile(r"(?:#|\bп\.|\bпункт\w*\s+|^\s*[,;])\s*(\d+(?:\.\d+)+)(?![\d.])", re.I)
POINT = re.compile(r'(?m)^\s*[«"]?(\d+(?:\.\d+)+)\.\s')


def render_citations(answer, mapping):
    def resolve(match):
        marker = match.group(1) or match.group(2)
        identifiers = list(IDENTIFIER.finditer(marker))
        citations = []
        for index, identifier in enumerate(identifiers):
            chunk = mapping.get("E" + identifier.group(1).lstrip("0"))
            if chunk is None:
                continue
            end = identifiers[index + 1].start() if index + 1 < len(identifiers) else len(marker)
            anchor = chunk.section
            point = EXPLICIT_POINT.search(marker[identifier.end():end])
            if chunk.category == "company" and point and point.group(1) in POINT.findall(chunk.text):
                anchor = "п. " + point.group(1)
            citations.append(f"{chunk.title}, {anchor}")
        return "[" + "; ".join(dict.fromkeys(citations)) + "]" if citations else ""

    return REFERENCE.sub(resolve, answer)
