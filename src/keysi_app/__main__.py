"""Command-line entry point for the KeySI Dash application."""

import os

from .application import app


def main() -> None:
    os.environ.setdefault("FLASK_ENV", "development")
    app.run(
        debug=os.getenv("KEYSI_DEBUG", "1").lower() in {"1", "true", "yes"},
        host=os.getenv("KEYSI_HOST", "127.0.0.1"),
        port=int(os.getenv("KEYSI_PORT", "47983")),
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
