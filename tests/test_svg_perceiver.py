from __future__ import annotations

from types import SimpleNamespace

from chart_agent.perception.svg_perceiver import perceive_svg


SVG_SAMPLE = """\
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <g id="axes_1">
    <g id="matplotlib.axis_1">
      <g id="xtick_1"><use x="10" y="30" /><text>1968</text></g>
      <g id="xtick_2"><use x="20" y="30" /><text>1969</text></g>
    </g>
    <g id="matplotlib.axis_2">
      <g id="ytick_1"><use x="0" y="20" /><text>0</text></g>
      <g id="ytick_2"><use x="0" y="10" /><text>1</text></g>
    </g>
    <g id="line2d_1">
      <path d="M 10 20 L 20 10" style="fill: none; stroke: #1f77b4" />
    </g>
  </g>
</svg>
"""


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.content)


def test_perceive_svg_rules_mode_preserves_existing_path(tmp_path, monkeypatch) -> None:
    svg_path = tmp_path / "sample.svg"
    svg_path.write_text(SVG_SAMPLE, encoding="utf-8")
    monkeypatch.delenv("SVG_PERCEPTION_MODE", raising=False)
    llm = FakeLLM('{"chart_type":"line"}')

    result = perceive_svg(str(svg_path), question="How many intersections?", llm=llm)

    assert result["perception_mode"] == "rules"
    assert result["chart_type"] == "line"
    assert result["mapping_info"]["llm_meta"]["mode"] == "rules"
    assert "Primitives:" in llm.prompts[0]


def test_perceive_svg_llm_summary_mode_uses_compact_svg_summary(tmp_path, monkeypatch) -> None:
    svg_path = tmp_path / "sample.svg"
    svg_path.write_text(SVG_SAMPLE, encoding="utf-8")
    monkeypatch.setenv("SVG_PERCEPTION_MODE", "llm")
    llm = FakeLLM('{"chart_type":"line"}')

    result = perceive_svg(str(svg_path), question="How many intersections?", llm=llm)

    assert result["perception_mode"] == "llm_summary"
    assert result["chart_type"] == "line"
    assert result["mapping_info"]["llm_meta"]["mode"] == "llm_summary"
    assert "SVG Summary:" in llm.prompts[0]
    assert "chart_type_guess" in llm.prompts[0]
