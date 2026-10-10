from django.conf import settings
from django.db import models


class AgentRun(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="agent_runs",
    )
    thread_id = models.UUIDField(unique=True, editable=False)
    task = models.TextField()
    repo_url = models.CharField(max_length=300, blank=True, default="")
    status = models.CharField(max_length=30, default="running")
    pr_url = models.CharField(max_length=500, blank=True, default="")
    summary = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.username} - {self.thread_id}"


class GitHubInstallation(models.Model):
    """A GitHub App installation a DevPilot user has proven access to."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="github_installations",
    )
    installation_id = models.BigIntegerField()
    account_login = models.CharField(max_length=100, blank=True, default="")
    repository_selection = models.CharField(max_length=20, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "installation_id")
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user} -> {self.account_login or self.installation_id}"
