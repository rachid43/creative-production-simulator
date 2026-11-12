"""Headless smoke tests: every page of both apps renders without exceptions."""
import os

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMPLE = ["Overview", "PODs", "Waste & Flow", "What-if"]
ADVANCED = ["Executive Command Center", "POD Production Network", "Mura Analysis", "Muri Analysis", "Muda Analysis",
            "Takt & Flow Performance", "Heijunka Workload Balancing", "Scenario Laboratory", "Improvement Recommendations"]


@pytest.mark.parametrize("script,pages", [("app.py", SIMPLE), ("app_advanced.py", ADVANCED)])
def test_all_pages_render(script, pages):
    at = AppTest.from_file(os.path.join(ROOT, script), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    for p in pages:
        at.session_state["page"] = p
        at.run()
        assert not at.exception, (p, [e.value for e in at.exception])


def test_invalid_input_is_reported_not_crashed():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300)
    at.run()
    at.session_state["simple"] = 60
    at.session_state["complex"] = 50
    at.run()
    assert not at.exception and any("exceed 100" in e.value for e in at.error)
