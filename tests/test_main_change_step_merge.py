from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


def _install_main_import_stubs() -> None:
    mods = {
        "langgraph": types.ModuleType("langgraph"),
        "langgraph.graph": types.ModuleType("langgraph.graph"),
        "chart_agent.core.perception_graph": types.ModuleType("chart_agent.core.perception_graph"),
        "chart_agent.core.answerer": types.ModuleType("chart_agent.core.answerer"),
        "chart_agent.core.vision_tool_phase": types.ModuleType("chart_agent.core.vision_tool_phase"),
        "chart_agent.llm_factory": types.ModuleType("chart_agent.llm_factory"),
        "chart_agent.perception.scatter_svg_updater": types.ModuleType("chart_agent.perception.scatter_svg_updater"),
        "chart_agent.perception.area_svg_updater": types.ModuleType("chart_agent.perception.area_svg_updater"),
        "chart_agent.perception.line_svg_updater": types.ModuleType("chart_agent.perception.line_svg_updater"),
        "chart_agent.perception.render_validator": types.ModuleType("chart_agent.perception.render_validator"),
        "chart_agent.perception.svg_perceiver": types.ModuleType("chart_agent.perception.svg_perceiver"),
    }
    mods["langgraph.graph"].END = object()

    class DummyStateGraph:
        def __init__(self, *args, **kwargs):
            pass

    mods["langgraph.graph"].StateGraph = DummyStateGraph
    mods["langgraph"].graph = mods["langgraph.graph"]
    mods["chart_agent.core.perception_graph"].run_perception = lambda inputs: None
    mods["chart_agent.core.answerer"].answer_question = lambda **kwargs: {}
    mods["chart_agent.core.vision_tool_phase"].run_visual_tool_phase = lambda **kwargs: {}
    mods["chart_agent.llm_factory"].make_llm = lambda *args, **kwargs: None
    mods["chart_agent.perception.scatter_svg_updater"].update_scatter_svg = lambda *args, **kwargs: None
    mods["chart_agent.perception.area_svg_updater"].update_area_svg = lambda *args, **kwargs: None
    mods["chart_agent.perception.area_svg_updater"].SVG_NS = "http://www.w3.org/2000/svg"
    mods["chart_agent.perception.line_svg_updater"].update_line_svg = lambda *args, **kwargs: None
    mods["chart_agent.perception.render_validator"].validate_render = lambda *args, **kwargs: {}
    mods["chart_agent.perception.svg_perceiver"].perceive_svg = lambda *args, **kwargs: {}

    for name, mod in mods.items():
        sys.modules.setdefault(name, mod)


class MainChangeStepMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_main_import_stubs()
        src_root = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(src_root))
        cls.main = importlib.import_module("main")

    def test_change_steps_keep_distinct_atomic_payloads_after_merge(self) -> None:
        structured_context = {
            "operation_target": {},
            "data_change": {
                "add": {
                    "category_name": "Ironclad Ventures",
                    "years": ["2015", "2016"],
                    "values": [0.58, 0.56],
                },
                "change": {
                    "changes": [
                        {
                            "category_name": "Aegis Corp",
                            "years": ["2021", "2022"],
                            "values": [0.3, 0.53],
                        },
                        {
                            "category_name": "Starlight Innovations",
                            "years": ["2022", "2023"],
                            "values": [0.51, 0.27],
                        },
                    ]
                },
            },
        }
        operation_plan = {
            "steps": [
                {
                    "operation": "add",
                    "operation_target": {"add_category": "Ironclad Ventures", "element_type": "area"},
                    "data_change": {
                        "add": {
                            "category_name": "Ironclad Ventures",
                            "years": ["2015", "2016"],
                            "values": [0.58, 0.56],
                        }
                    },
                },
                {
                    "operation": "change",
                    "operation_target": {"category_name": "Aegis Corp", "element_type": "area"},
                    "data_change": {
                        "change": {
                            "category_name": "Aegis Corp",
                            "years": ["2021", "2022"],
                            "values": [0.3, 0.53],
                        }
                    },
                },
                {
                    "operation": "change",
                    "operation_target": {"category_name": "Starlight Innovations", "element_type": "area"},
                    "data_change": {
                        "change": {
                            "category_name": "Starlight Innovations",
                            "years": ["2022", "2023"],
                            "values": [0.51, 0.27],
                        }
                    },
                },
            ]
        }

        steps = self.main._operation_steps_from_plan(
            operation_plan,
            "Add Ironclad Ventures and apply listed value revisions.",
            structured_context,
        )
        rendered = [self.main._render_structured_step_question(step) for step in steps]

        self.assertEqual(len(steps), 5)
        self.assertIn('Change "Starlight Innovations" in 2022 to 0.51', rendered)
        self.assertIn('Change "Starlight Innovations" in 2023 to 0.27', rendered)
        self.assertEqual(
            steps[3]["data_change"],
            {"changes": [{"category_name": "Starlight Innovations", "years": ["2022"], "values": [0.51]}]},
        )
        self.assertEqual(
            steps[4]["data_change"],
            {"changes": [{"category_name": "Starlight Innovations", "years": ["2023"], "values": [0.27]}]},
        )


if __name__ == "__main__":
    unittest.main()
