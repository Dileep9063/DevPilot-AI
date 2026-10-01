from django.conf import settings
from django.db import models


class AgentRun(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="agent_runs")
    thread_id = models.UUIDField(unique=True, editable=False)
    task = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{this.user.username} - {this.thread_id}"
