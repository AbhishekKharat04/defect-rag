"""CLI entrypoint for the defect-rag package.

Provides ``defect-rag serve`` and ``defect-rag index`` commands so the
application can be launched without memorising ``uvicorn`` invocations.
"""

import argparse
import sys


def main() -> None:
    """Parse top-level sub-commands and dispatch to the appropriate handler."""
    parser = argparse.ArgumentParser(
        prog="defect-rag",
        description="Vision-Language RAG Assistant for industrial defect detection.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- serve -----------------------------------------------------------
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI backend server.")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address (default: 0.0.0.0).")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")

    # --- index -----------------------------------------------------------
    index_parser = subparsers.add_parser("index", help="Index a dataset category into Qdrant.")
    index_parser.add_argument("--category", type=str, default="bottle", help="MVTec category name (default: bottle).")
    index_parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic dataset instead of downloading.")
    index_parser.add_argument("--recreate", action="store_true", help="Recreate the Qdrant collection before indexing.")

    # --- frontend --------------------------------------------------------
    subparsers.add_parser("frontend", help="Launch the Gradio frontend UI.")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    elif args.command == "index":
        from scripts.index_dataset import main as index_main
        sys.argv = [
            "index_dataset",
            *(["--recreate"] if args.recreate else []),
            *(["--synthetic"] if args.synthetic else []),
            "--category", args.category,
        ]
        index_main()
    elif args.command == "frontend":
        from src.frontend.app import demo
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
