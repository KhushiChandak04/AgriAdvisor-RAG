"""Send a synthetic run to confirm LangSmith configuration and connectivity."""
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from src import config


def main() -> None:
    if not config.LANGSMITH_API_KEY or config.LANGSMITH_API_KEY.startswith("your_"):
        raise SystemExit("Set LANGSMITH_API_KEY in the project-root .env file first.")

    project = config.LANGSMITH_PROJECT
    client = Client(
        api_key=config.LANGSMITH_API_KEY,
        api_url=config.LANGSMITH_ENDPOINT or None,
    )
    if not list(client.list_projects(name=project, limit=1)):
        raise SystemExit(
            f"LangSmith project {project!r} was not found or is not accessible to this API key."
        )

    run_id = uuid.uuid4()
    client.create_run(
        name="agriadvisor_connectivity_check",
        run_type="chain",
        inputs={"check": "synthetic connectivity test"},
        project_name=project,
        id=run_id,
    )
    client.update_run(
        run_id,
        outputs={"status": "connected"},
        end_time=datetime.now(timezone.utc),
    )
    client.flush()
    client.read_run(run_id)
    print(f"LangSmith connectivity verified. Project: {project}. Synthetic run: {run_id}")


if __name__ == "__main__":
    main()
