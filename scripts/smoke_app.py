"""Exercise key dashboard filters without launching a public listener."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def main() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45).run()
    if app.exception:
        raise AssertionError("Dashboard raised during startup: " + str(app.exception[0].message))
    if not app.selectbox or len(app.selectbox) < 3:
        raise AssertionError("Dashboard did not render its channel, label, and segment selectors")
    first_channel = next(value for value in app.selectbox[0].options if value != "All channels")
    app.selectbox[0].set_value(first_channel).run()
    if app.exception:
        raise AssertionError("Channel filter interaction failed: " + str(app.exception[0].message))
    app.selectbox[1].set_value("Anomalies").run()
    if app.exception:
        raise AssertionError(
            "Reference-label filter interaction failed: " + str(app.exception[0].message)
        )
    print("Streamlit startup, channel filter, and anomaly filter checks passed")


if __name__ == "__main__":
    main()
