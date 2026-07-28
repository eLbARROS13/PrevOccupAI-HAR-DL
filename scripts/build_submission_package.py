#!/usr/bin/env python3
"""Build a flat, checksummed Springer Nature LaTeX submission package."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import zipfile
from pathlib import Path


INPUT_PATTERN = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
GRAPHICS_PATTERN = re.compile(
    r"(?P<prefix>\\includegraphics(?:\[[^]]*\])?\s*\{)(?P<path>[^}]+)(?P<suffix>\})"
)
GRAPHICS_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps", ".tif", ".tiff")
REQUIRED_SUPPORT_FILES = ("sn-jnl.cls", "sn-basic.bst", "references.bib")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Flatten a modular manuscript and create a submission ZIP."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("extended_mban_har_paper"),
        help="Directory containing the modular main.tex.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extended_mban_har_paper/build/submission_package"),
        help="Flat output directory.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("extended_mban_har_paper/build/submission_package.zip"),
        help="Output ZIP path.",
    )
    return parser.parse_args()


def ensure_within_source(path: Path, source_dir: Path) -> Path:
    """Resolve a path and reject references outside the manuscript source tree."""
    resolved = path.resolve()
    try:
        resolved.relative_to(source_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Referenced path escapes source directory: {path}") from exc
    return resolved


def resolve_tex_reference(reference: str, parent: Path, source_dir: Path) -> Path:
    """Resolve an input relative to the current file or main compilation root."""
    candidates = [parent / reference, source_dir / reference]
    checked: list[Path] = []
    for candidate in candidates:
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".tex")
        candidate = ensure_within_source(candidate, source_dir)
        checked.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing LaTeX input; checked: {checked}")


def expand_tex_file(path: Path, source_dir: Path, stack: tuple[Path, ...] = ()) -> str:
    """Recursively expand LaTeX input/include directives."""
    resolved = ensure_within_source(path, source_dir)
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"Cyclic LaTeX input detected: {cycle}")

    text = resolved.read_text(encoding="utf-8")

    def replace_input(match: re.Match[str]) -> str:
        child = resolve_tex_reference(match.group(1).strip(), resolved.parent, source_dir)
        relative_child = child.relative_to(source_dir.resolve())
        expanded = expand_tex_file(child, source_dir, (*stack, resolved))
        return f"\n% BEGIN expanded input: {relative_child}\n{expanded}\n% END expanded input: {relative_child}\n"

    return INPUT_PATTERN.sub(replace_input, text)


def resolve_graphic(reference: str, source_dir: Path) -> Path:
    """Resolve a graphics reference, including extension-free references."""
    base = ensure_within_source(source_dir / reference, source_dir)
    candidates = (base,) if base.suffix else tuple(base.with_suffix(ext) for ext in GRAPHICS_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing figure referenced by manuscript: {reference}")


def flatten_graphics(text: str, source_dir: Path, output_dir: Path) -> str:
    """Copy referenced graphics to the package root and rewrite their paths."""
    copied_sources: dict[str, Path] = {}

    def replace_graphic(match: re.Match[str]) -> str:
        reference = match.group("path").strip()
        source = resolve_graphic(reference, source_dir)
        basename = source.name
        previous = copied_sources.get(basename)
        if previous is not None and previous != source:
            raise ValueError(
                f"Figure basename collision during flattening: {previous} and {source}"
            )
        copied_sources[basename] = source
        shutil.copy2(source, output_dir / basename)
        return f"{match.group('prefix')}{basename}{match.group('suffix')}"

    return GRAPHICS_PATTERN.sub(replace_graphic, text)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output_dir: Path) -> Path:
    """Write hashes for all payload files except the manifest itself."""
    manifest = output_dir / "MANIFEST.sha256"
    payload_files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path != manifest
    )
    lines = [f"{sha256(path)}  {path.name}" for path in payload_files]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def write_archive(output_dir: Path, archive_path: Path) -> None:
    """Create a deterministic-name ZIP archive from the flat package."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)


def build_package(source_dir: Path, output_dir: Path, archive_path: Path) -> None:
    """Build the flat source directory, manifest, and ZIP archive."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    archive_path = archive_path.resolve()
    main_tex = source_dir / "main.tex"
    if not main_tex.is_file():
        raise FileNotFoundError(f"Missing manuscript entry point: {main_tex}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    expanded = expand_tex_file(main_tex, source_dir)
    expanded = flatten_graphics(expanded, source_dir, output_dir)
    (output_dir / "main.tex").write_text(expanded, encoding="utf-8")

    for filename in REQUIRED_SUPPORT_FILES:
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing required support file: {source}")
        shutil.copy2(source, output_dir / filename)

    remaining_inputs = INPUT_PATTERN.findall(expanded)
    if remaining_inputs:
        raise ValueError(f"Unexpanded LaTeX inputs remain: {remaining_inputs}")

    write_manifest(output_dir)
    write_archive(output_dir, archive_path)


def main() -> None:
    """Run the package builder."""
    args = parse_args()
    build_package(args.source_dir, args.output_dir, args.archive)
    print(f"Flat package: {args.output_dir.resolve()}")
    print(f"ZIP archive: {args.archive.resolve()}")


if __name__ == "__main__":
    main()
