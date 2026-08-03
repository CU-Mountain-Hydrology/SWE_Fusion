# run/checkpoint_manager.py

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Tracks which pipeline steps have completed for a given model run date,persisted to disk as JSON, so run_model() can
    resume mid-pipeline after a fatal error (e.g. an OOM kill) instead of restarting from scratch.

    TODO: pickle local variables instead of serializing them into the JSON
    """

    def __init__(self, date: int, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{date}.json"
        self.state = self._load()

    def _load(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {"completed_steps": [], "results": {}, "last_updated": None}

    def _save(self):
        self.state["last_updated"] = datetime.now().isoformat()
        # write-then-rename so a crash mid-write can't corrupt the checkpoint
        tmp = self.checkpoint_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2)
        tmp.replace(self.checkpoint_file)

    def is_done(self, step_name: str) -> bool:
        return step_name in self.state["completed_steps"]

    def mark_done(self, step_name: str, result=None):
        if step_name not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step_name)
        if result is not None:
            self.state["results"][step_name] = result
        self._save()

    def get_result(self, step_name: str):
        return self.state["results"].get(step_name)

    def reset(self):
        # Wipe checkpoint state to force a full rerun.
        self.state = {"completed_steps": [], "results": {}, "last_updated": None}
        self._save()

    def run_step(self, step_name: str, func, *args, **kwargs):
        # Run func unless it's already marked complete for this date.
        if self.is_done(step_name):
            logger.info(f"Skipping '{step_name}' (already completed).")
            return self.get_result(step_name)
        logger.info(f"Running '{step_name}'...")
        result = func(*args, **kwargs)
        self.mark_done(step_name, result)
        logger.info(f"Completed '{step_name}'.")
        return result

    def archive(self):
        # Move the checkpoint file to a `completed/` subdirectory once a run finishes successfully.
        if not self.checkpoint_file.exists():
            logger.warning(f"No checkpoint file to archive at {self.checkpoint_file}.")
            return
        archive_dir = self.checkpoint_dir / "completed"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_path = archive_dir / self.checkpoint_file.name
        self.checkpoint_file.replace(archived_path)
        logger.info(f"Archived checkpoint to {archived_path}.")