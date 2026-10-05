import os
import shutil
import stat
import time
import uuid

from django.core.management.base import BaseCommand

from core import agent_service
from core.models import AgentRun

# Runs in these states may still be resumed, so their clones are kept.
ACTIVE_STATUSES = {"running", "waiting_for_approval"}


def _force_remove(func, path, _exc):
    """Windows: git object files are read-only; clear the flag and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


class Command(BaseCommand):
    help = (
        "Delete cloned repository workspaces (workspace/runs/<thread_id>) "
        "older than --days. Runs that are still waiting for approval are kept."
    )

    def add_arguments(self, parser):
        parser.add_argument("--days", type=float, default=7.0)
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        runs_dir = agent_service.WORKSPACE_ROOT / "runs"

        if not runs_dir.exists():
            self.stdout.write("No workspaces to clean.")
            return

        cutoff = time.time() - options["days"] * 86400
        removed = kept = 0

        for entry in sorted(runs_dir.iterdir()):
            if not entry.is_dir():
                continue

            try:
                thread_id = uuid.UUID(entry.name)
            except ValueError:
                continue  # not one of ours

            if entry.stat().st_mtime > cutoff:
                kept += 1
                continue

            status = (
                AgentRun.objects.filter(thread_id=thread_id)
                .values_list("status", flat=True)
                .first()
            )
            if status in ACTIVE_STATUSES:
                kept += 1
                continue

            self.stdout.write(f"{'Would remove' if options['dry_run'] else 'Removing'} {entry}")
            if not options["dry_run"]:
                shutil.rmtree(entry, onerror=_force_remove)
            removed += 1

        self.stdout.write(self.style.SUCCESS(f"Done: {removed} removed, {kept} kept."))
