"""Streamlit Community Cloud entry point: streamlit run app.py."""

from asl_recognition.config import configure_runtime


def main():
    configure_runtime()
    # Configure native thread pools before importing numerical/video libraries.
    from asl_recognition.ui import main as render_app

    render_app()


if __name__ == "__main__":
    main()
