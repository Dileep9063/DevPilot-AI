from django.contrib import admin
from django.http import JsonResponse
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from core.auth_views import RegisterView
from core.views import (
    status, ai_chat, agent_run, agent_resume, agent_runs, agent_run_detail,
    github_status, github_repositories, github_connect, github_disconnect, github_callback,
)


def home(request):
    return JsonResponse({"message": "Welcome to DevPilot AI API"})


urlpatterns = [
    path("", home),
    path("admin/", admin.site.urls),
    path("api/status/", status),
    path("api/auth/register/", RegisterView.as_view()),
    path("api/auth/login/", TokenObtainPairView.as_view()),
    path("api/auth/refresh/", TokenRefreshView.as_view()),
    path("api/ai/chat/", ai_chat),
    path("api/agent/run/", agent_run),
    path("api/agent/resume/", agent_resume),
    path("api/agent/runs/", agent_runs),
    path("api/github/status/", github_status),
    path("api/github/repositories/", github_repositories),
    path("api/github/connect/", github_connect),
    path("api/github/disconnect/", github_disconnect),
    path("api/github/callback/", github_callback),
    path("api/agent/runs/<uuid:thread_id>/", agent_run_detail),
]
