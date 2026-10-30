"""Command line interface: `garage-copilot --help`."""

from __future__ import annotations

import argparse
import json
import sys

from garage_copilot.config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="garage-copilot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Train the MOT risk model on synthetic data and save it")
    p_vehicle = sub.add_parser("vehicle", help="Look up a vehicle")
    p_vehicle.add_argument("registration")
    p_risk = sub.add_parser("risk", help="Predict MOT failure risk")
    p_risk.add_argument("registration")
    p_ask = sub.add_parser("ask", help="Ask the knowledge base a question")
    p_ask.add_argument("question")
    p_agent = sub.add_parser("agent", help="Talk to the agent")
    p_agent.add_argument("message")
    p_eval = sub.add_parser("eval", help="Measure retrieval quality (hit@k, MRR)")
    p_eval.add_argument("--k", type=int, default=3)
    p_serve = sub.add_parser("serve", help="Run the API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "train":
        from garage_copilot.ml.model import MotRiskModel

        model = MotRiskModel.train_default()
        model.save(settings.model_path)
        print(f"Saved model to {settings.model_path}")
        print(json.dumps(model.metrics, indent=2))
        return 0

    if args.command == "eval":
        from garage_copilot.config import DATA_DIR
        from garage_copilot.rag.assistant import RagAssistant
        from garage_copilot.rag.evaluate import evaluate_retrieval, load_questions

        rag = RagAssistant.from_directory(settings.knowledge_dir)
        report = evaluate_retrieval(
            rag.store, load_questions(DATA_DIR / "eval" / "rag_questions.jsonl"), k=args.k
        )
        print(json.dumps(report.as_dict(), indent=2))
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("garage_copilot.api:create_app", factory=True, host=args.host, port=args.port)
        return 0

    from garage_copilot.core import Copilot

    copilot = Copilot.build(settings)
    if args.command == "vehicle":
        print(copilot.lookup.get(args.registration).model_dump_json(indent=2))
    elif args.command == "risk":
        score = copilot.risk_model.score(copilot.lookup.get(args.registration))
        print(json.dumps(score.__dict__, indent=2))
    elif args.command == "ask":
        answer = copilot.rag.ask(args.question)
        print(answer.text)
        for c in answer.citations:
            print(f"  - {c}")
    elif args.command == "agent":
        result = copilot.agent.run(args.message)
        print(result.reply)
        print(f"\n(tools used: {', '.join(result.tools_used) or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
