"""
GitHub App integration (per-user repository access).

Flow
----
1. A logged-in user clicks "Connect GitHub"  -> connect_url(state)
2. They install the App on their account and pick repositories.
3. GitHub redirects the browser to our callback with ?code & installation_id.
4. We exchange the code for a *user* token, ask GitHub which installations that
   user can access, and store the installation ONLY if it is in that list.
   (Otherwise anyone could claim someone else's installation_id.)
5. For each run we resolve the repo's installation, check that it belongs to
   the user, and mint a short-lived *installation* token scoped to that single
   repository. Tokens are never stored: they are minted when needed.

Required environment (all of them, otherwise the App mode is off and the
single GITHUB_TOKEN from .env keeps being used):
    GITHUB_APP_ID, GITHUB_APP_SLUG, GITHUB_APP_CLIENT_ID,
    GITHUB_APP_CLIENT_SECRET, and one of
    GITHUB_APP_PRIVATE_KEY (PEM, newlines written as \\n) or
    GITHUB_APP_PRIVATE_KEY_PATH (path to the .pem file)
"""

import os
import time
from pathlib import Path
from urllib.parse import quote

import jwt
import requests
from django.core import signing

API = "https://api.github.com"
STATE_SALT = "devpilot.github.connect"
STATE_MAX_AGE = 600  # seconds

_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


class GitHubAccessError(Exception):
    """A problem the user can act on (not connected, not installed, ...)."""


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _private_key() -> str:
    pem = os.environ.get("GITHUB_APP_PRIVATE_KEY", "")
    if pem.strip():
        return pem.replace("\\n", "\n")

    path = _env("GITHUB_APP_PRIVATE_KEY_PATH")
    if path:
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def is_configured() -> bool:
    return bool(
        _env("GITHUB_APP_ID")
        and _env("GITHUB_APP_SLUG")
        and _env("GITHUB_APP_CLIENT_ID")
        and _env("GITHUB_APP_CLIENT_SECRET")
        and _private_key()
    )


def install_page_url() -> str:
    return f"https://github.com/apps/{_env('GITHUB_APP_SLUG')}/installations/new"


# ------------------------------------------------------------
# The "state" value that ties the browser round-trip to a user
# ------------------------------------------------------------

def make_state(user_id: int) -> str:
    return signing.dumps({"uid": user_id}, salt=STATE_SALT)


def read_state(state: str):
    """Return the user id inside a valid state, else None."""
    try:
        data = signing.loads(state, salt=STATE_SALT, max_age=STATE_MAX_AGE)
        return int(data["uid"])
    except (signing.BadSignature, KeyError, TypeError, ValueError):
        return None


def connect_url(state: str) -> str:
    return f"{install_page_url()}?state={quote(state)}"


# ------------------------------------------------------------
# GitHub API calls
# ------------------------------------------------------------

def app_jwt() -> str:
    """Short-lived JWT that authenticates as the App itself."""
    now = int(time.time())
    return jwt.encode(
        {"iat": now - 60, "exp": now + 9 * 60, "iss": _env("GITHUB_APP_ID")},
        _private_key(),
        algorithm="RS256",
    )


def _app_headers() -> dict:
    return {**_HEADERS, "Authorization": f"Bearer {app_jwt()}"}


def exchange_code(code: str) -> str:
    """Trade the OAuth ?code for a user token (used once, never stored)."""
    resp = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={
            "client_id": _env("GITHUB_APP_CLIENT_ID"),
            "client_secret": _env("GITHUB_APP_CLIENT_SECRET"),
            "code": code,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("access_token")

    if not token:
        raise GitHubAccessError(
            "GitHub did not accept the authorization code: "
            + str(data.get("error_description") or data.get("error") or "unknown error")
        )
    return token


def user_installations(user_token: str) -> list:
    """Installations of this App that the user can access."""
    resp = requests.get(
        f"{API}/user/installations",
        headers={**_HEADERS, "Authorization": f"Bearer {user_token}"},
        params={"per_page": 100},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("installations", [])



def installation_repositories(installation_id: int) -> list:
    """List repositories currently accessible to one connected installation."""
    token = installation_token(installation_id)
    repositories = []
    page = 1
    while True:
        resp = requests.get(
            f"{API}/installation/repositories",
            headers={**_HEADERS, "Authorization": f"Bearer {token}"},
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json().get("repositories", [])
        repositories.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return repositories


def find_repo_installation(owner: str, repo: str):
    """Installation id that covers owner/repo, or None if the App isn't installed there."""
    resp = requests.get(
        f"{API}/repos/{owner}/{repo}/installation",
        headers=_app_headers(),
        timeout=30,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()["id"]


def installation_token(installation_id: int, repo_name: str | None = None) -> str:
    """Mint a ~1 hour token, limited to one repository when repo_name is given."""
    resp = requests.post(
        f"{API}/app/installations/{installation_id}/access_tokens",
        headers=_app_headers(),
        json={"repositories": [repo_name]} if repo_name else {},
        timeout=30,
    )

    if resp.status_code == 404:
        raise GitHubAccessError(
            "The DevPilot GitHub App is no longer installed for this account. "
            "Click 'Connect GitHub' to install it again."
        )
    if resp.status_code == 422:
        raise GitHubAccessError(
            f"The DevPilot GitHub App does not have access to '{repo_name}'. "
            "Add the repository in your GitHub App installation settings."
        )

    resp.raise_for_status()
    return resp.json()["token"]


# ------------------------------------------------------------
# Authorisation used when a run starts
# ------------------------------------------------------------

def resolve_repo_access(user, owner: str, repo: str) -> int:
    """
    Return the installation id to use for owner/repo, or raise
    GitHubAccessError explaining what the user must do.
    """
    from core.models import GitHubInstallation

    mine = GitHubInstallation.objects.filter(user=user)

    if not mine.exists():
        raise GitHubAccessError(
            "Connect your GitHub account first (use the 'Connect GitHub' button), "
            "then choose which repositories DevPilot may access."
        )

    installation_id = find_repo_installation(owner, repo)

    if installation_id is None:
        raise GitHubAccessError(
            f"DevPilot is not installed on {owner}/{repo}. Add this repository to "
            f"the DevPilot GitHub App ({install_page_url()}) and try again."
        )

    if not mine.filter(installation_id=installation_id).exists():
        raise GitHubAccessError(
            f"{owner}/{repo} belongs to a GitHub installation that is not connected "
            "to your DevPilot account. Use 'Connect GitHub' with the account that "
            "owns this repository."
        )

    return installation_id
