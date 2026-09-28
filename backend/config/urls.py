from django.contrib import admin
from django.urls import path
from django.http import JsonResponse

from core.views import status, ai_chat, agent_run, agent_resume


def home(request):
    return JsonResponse({
        "message": "Welcome to DevPilot AI API"
    })


urlpatterns = [
    path("", home),
    path("admin/", admin.site.urls),
    path("api/status/", status),
    path("api/ai/chat/", ai_chat),
    path("api/agent/run/", agent_run),
    path("api/agent/resume/", agent_resume),
]