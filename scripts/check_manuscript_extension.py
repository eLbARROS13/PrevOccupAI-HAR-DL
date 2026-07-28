#!/usr/bin/env python3
"""Produce a reproducible, non-substantive textual-extension diagnostic."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


INPUT_PATTERN = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
COMMAND_PATTERN = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?")
BRACE_PATTERN = re.compile(r"[{}$]")
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare conference and journal text. The output measures textual change "
            "only and cannot establish a substantive scientific extension."
        )
    )
    parser.add_argument("conference", type=Path, help="Conference PDF or text file.")
    parser.add_argument("journal", type=Path, help="Journal PDF, TeX, or text file.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--shingle-size", type=int, default=8, help="Word shingle size.")
    return parser.parse_args()


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using Poppler's pdftotext executable."""
    executable = shutil.which("pdftotext")
    if executable is None:
        raise RuntimeError("pdftotext is required to inspect PDF input")
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "document.txt"
        subprocess.run(
            [executable, "-layout", str(path), str(output_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return output_path.read_text(encoding="utf-8", errors="replace")


def expand_tex(
    path: Path,
    stack: tuple[Path, ...] = (),
    project_root: Path | None = None,
) -> str:
    """Expand local LaTeX input/include directives for text analysis."""
    resolved = path.resolve()
    project_root = project_root or resolved.parent
    if resolved in stack:
        raise ValueError(f"Cyclic LaTeX input: {resolved}")
    text = resolved.read_text(encoding="utf-8")

    def replace_input(match: re.Match[str]) -> str:
        reference = match.group(1).strip()
        candidates = [resolved.parent / reference, project_root / reference]
        for child in candidates:
            if child.suffix == "":
                child = child.with_suffix(".tex")
            if child.is_file():
                return expand_tex(child, (*stack, resolved), project_root)
        raise FileNotFoundError(f"Missing LaTeX input; checked: {candidates}")

    return INPUT_PATTERN.sub(replace_input, text)


def read_document(path: Path) -> str:
    """Read supported document formats."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix == ".tex":
        return expand_tex(path)
    return path.read_text(encoding="utf-8", errors="replace")


def normalise_words(text: str) -> list[str]:
    """Convert source text into case-normalised word tokens."""
    text = re.sub(r"(?m)(?<!\\)%.*$", " ", text)
    text = COMMAND_PATTERN.sub(" ", text)
    text = BRACE_PATTERN.sub(" ", text)
    return [token.casefold() for token in WORD_PATTERN.findall(text)]


def make_shingles(words: list[str], size: int) -> Counter[tuple[str, ...]]:
    """Return a multiset of consecutive word shingles."""
    if size < 1:
        raise ValueError("Shingle size must be at least one")
    return Counter(tuple(words[index : index + size]) for index in range(len(words) - size + 1))


def compare_documents(conference_words: list[str], journal_words: list[str], size: int) -> dict[str, object]:
    """Calculate transparent word-count and shingle-overlap diagnostics."""
    conference_shingles = make_shingles(conference_words, size)
    journal_shingles = make_shingles(journal_words, size)
    shared = conference_shingles & journal_shingles
    union = conference_shingles | journal_shingles
    shared_count = sum(shared.values())
    journal_shingle_count = sum(journal_shingles.values())
    conference_shingle_count = sum(conference_shingles.values())
    union_count = sum(union.values())
    word_growth = len(journal_words) - len(conference_words)

    return {
        "conference_word_count": len(conference_words),
        "journal_word_count": len(journal_words),
        "net_word_growth": word_growth,
        "net_word_growth_percent_of_conference": (
            100.0 * word_growth / len(conference_words) if conference_words else None
        ),
        "shingle_size_words": size,
        "conference_shingle_count": conference_shingle_count,
        "journal_shingle_count": journal_shingle_count,
        "shared_shingle_count": shared_count,
        "journal_shingle_overlap_percent": (
            100.0 * shared_count / journal_shingle_count if journal_shingle_count else None
        ),
        "conference_shingle_coverage_percent": (
            100.0 * shared_count / conference_shingle_count if conference_shingle_count else None
        ),
        "multiset_shingle_jaccard_percent": (
            100.0 * shared_count / union_count if union_count else None
        ),
        "interpretation_warning": (
            "These are textual diagnostics only. They do not distinguish paraphrase, "
            "reorganisation, references, placeholders, or genuinely new scientific content "
            "and cannot establish compliance with a substantive-extension threshold."
        ),
    }


def main() -> None:
    """Run the comparison and emit JSON."""
    args = parse_args()
    conference_words = normalise_words(read_document(args.conference))
    journal_words = normalise_words(read_document(args.journal))
    result = compare_documents(conference_words, journal_words, args.shingle_size)
    result["conference_source"] = str(args.conference.resolve())
    result["journal_source"] = str(args.journal.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
