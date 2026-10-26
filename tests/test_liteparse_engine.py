"""The fast engine: one command, and an identity that changes when the toolchain does."""
from __future__ import annotations


from ocr_playground.domain.document import PageImage
from ocr_playground.infrastructure.engines import liteparse


class FakeRunner:
    def __init__(self, output: str = "  markdown out\n") -> None:
        self.output = output
        self.commands: list[list[str]] = []

    def run(self, command, timeout=None) -> str:
        self.commands.append(list(command))
        return self.output


def test_a_page_is_one_liteparse_call(tmp_path) -> None:
    runner = FakeRunner()
    engine = liteparse.LiteParseEngine(runner=runner, lit_path="/opt/lit")
    target = tmp_path / "page-1.png"

    assert engine.read(PageImage(number=1, path=target)) == "markdown out"
    assert runner.commands == [["/opt/lit", "parse", str(target), "--format", "markdown", "-q"]]


def test_the_engine_is_always_available_and_says_so_in_its_identity(monkeypatch) -> None:
    versions = {"lit": "lit 2.17.0", "tesseract": "tesseract 5.3.4"}
    monkeypatch.setattr(
        liteparse, "first_line", lambda command, fallback="unknown": versions[command[0]]
    )
    engine = liteparse.LiteParseEngine(runner=FakeRunner(), lit_path="lit")

    assert engine.available is True
    assert engine.identity() == (
        "liteparse-tesseract lit=lit 2.17.0 tesseract=tesseract 5.3.4"
    )


def test_a_missing_tool_degrades_to_unknown(monkeypatch) -> None:
    monkeypatch.setattr(liteparse, "first_line", lambda command, fallback="unknown": fallback)
    engine = liteparse.LiteParseEngine(runner=FakeRunner(), lit_path="lit")
    assert engine.identity() == "liteparse-tesseract lit=unknown tesseract=unknown"
