"""Tests for custom immutable source scopes."""

from pathlib import Path

from prevoccupai_har.source_snapshot import (
    build_source_tree_manifest,
    load_source_tree_manifest,
    write_source_tree_manifest,
)


def test_custom_source_scope_ignores_explicitly_mutable_policy(tmp_path: Path) -> None:
    source = tmp_path / "src" / "package"
    configs = tmp_path / "configs"
    source.mkdir(parents=True)
    configs.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (configs / "frozen.json").write_text('{"frozen": true}\n', encoding="utf-8")
    policy = configs / "mutable_policy.json"
    policy.write_text('{"enabled": false}\n', encoding="utf-8")
    patterns = ("src/**/*.py", "configs/frozen.json")
    manifest = build_source_tree_manifest(tmp_path, patterns=patterns)
    manifest_path = tmp_path / "manifest.json"
    write_source_tree_manifest(manifest_path, manifest)

    policy.write_text('{"enabled": true}\n', encoding="utf-8")

    loaded = load_source_tree_manifest(
        manifest_path,
        root=tmp_path,
        verify_current_tree=True,
    )
    assert loaded["patterns"] == list(patterns)
    assert all(
        value["relative_path"] != "configs/mutable_policy.json"
        for value in loaded["files"]
    )

