# GitHub App setup (one-time, ~10 minutes)

DevPilot can work in two modes:

| Mode | When | How it authenticates |
|---|---|---|
| **Single token** (default) | Only you use it | `GITHUB_TOKEN` in `.env`; you must grant that token access to every repo |
| **GitHub App** | Several users, or you don't want to edit a token per repo | Each user clicks **Connect GitHub** once and chooses which repos to share |

The mode switches on automatically: if all `GITHUB_APP_*` values below are set,
App mode is used; otherwise `GITHUB_TOKEN` keeps working exactly as before.

## 1. Register the App on GitHub

Only you can do this (it lives in your GitHub account).

1. GitHub → your profile picture → **Settings** → (bottom of the left menu)
   **Developer settings** → **GitHub Apps** → **New GitHub App**.
2. Fill in the form. Labels may differ slightly from GitHub's current wording.

| Field | Value |
|---|---|
| GitHub App name | `DevPilot AI <your name>` (must be unique on GitHub) |
| Homepage URL | `http://localhost:5173` (any URL is accepted) |
| **Callback URL** | `http://localhost:8000/api/github/callback/` |
| **Request user authorization (OAuth) during installation** | ✅ **tick this** |
| Expire user authorization tokens | leave as is (DevPilot never stores user tokens) |
| Setup URL | leave empty (it is disabled when the box above is ticked) |
| **Webhook → Active** | ❌ **untick** (DevPilot does not need webhooks) |

3. **Repository permissions** (leave everything else as "No access"):

| Permission | Access |
|---|---|
| **Contents** | **Read and write** |
| **Pull requests** | **Read and write** |
| Metadata | Read-only (added automatically) |

4. **Where can this GitHub App be installed?**
   Choose **Only on this account** while testing. Switch to **Any account**
   later when other people should use it.
5. Click **Create GitHub App**.

## 2. Collect four values

On the App's settings page:

1. **App ID** — a number near the top.
2. **App slug** — the last part of its public link
   `https://github.com/apps/<slug>`.
3. **Client ID** — shown near the top (starts with `Iv`).
4. **Client secret** — click **Generate a new client secret** and copy it
   immediately. GitHub shows it only once.
5. **Private key** — scroll to *Private keys* → **Generate a private key**.
   A `.pem` file downloads. Move it **outside** the project folder, for
   example `C:\devpilot-keys\devpilot.pem`. Never commit it.

## 3. Configure DevPilot

Add to `backend\.env`:

```
GITHUB_APP_ID=123456
GITHUB_APP_SLUG=devpilot-ai-yourname
GITHUB_APP_CLIENT_ID=Iv23xxxxxxxxxxxx
GITHUB_APP_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxx
GITHUB_APP_PRIVATE_KEY_PATH=C:\devpilot-keys\devpilot.pem
FRONTEND_URL=http://localhost:5173
```

Then:

```powershell
cd backend
python manage.py migrate          # adds the GitHubInstallation table
python manage.py runserver        # restart so .env is re-read
```

`GITHUB_TOKEN` is no longer needed in App mode (it is only a fallback when the
App is not configured).

### Docker

Put the same variables in the root `.env`. A `.pem` file cannot be a
multi-line value there, so write the key on **one line with literal `\n`**
between lines, **without quotes**:

```powershell
(Get-Content C:\devpilot-keys\devpilot.pem -Raw).Trim() -replace "`r?`n", "\n"
```

Use the output as `GITHUB_APP_PRIVATE_KEY=...` (instead of `..._PATH`) and set
`FRONTEND_URL=http://localhost:8080`.

## 4. Connect and test

1. Open the DevPilot UI and log in. A **Connect GitHub** card appears.
2. Click it. On GitHub choose your account, pick **Only select repositories**,
   choose a throwaway repo, then **Install & Authorize**.
3. You are sent back to DevPilot with *"GitHub connected"*.
4. Paste that repo's URL, run a task, approve the diff and the push.
5. Check that the pull request was opened by **`<your app name>[bot]`**.

To give access to another repo later, click **Add or change repositories** and
tick it. No token editing, no restart.

## How it works (and why it is safe)

- Each user's **Connect GitHub** result is saved only after GitHub confirms,
  using that user's own authorization, that they can access the installation.
  Pasting someone else's installation id does not work.
- When a run starts, DevPilot finds the installation that covers the repo and
  requires it to be **one this user connected**. Another user's repo is refused
  even if the App is installed there.
- Tokens are minted per use, last about an hour, and are limited to the single
  repository of the run. They are **not** stored in the database, the run
  checkpoints, or `.git/config`.
- Private repositories can be cloned (the previous anonymous clone could not).

## For production

- Use HTTPS URLs. Add them as extra **Callback URLs** on the App (GitHub allows
  several) and set `FRONTEND_URL`, `CORS_ALLOWED_ORIGINS` and
  `CSRF_TRUSTED_ORIGINS` to match.
- Keep the private key in a secret manager or an environment variable, not in
  a file in the repository. If it leaks, generate a new one on the App page and
  delete the old one.
- DevPilot does not listen for webhooks, so if a user uninstalls the App on
  GitHub the next run fails with a message asking them to reconnect.

## Troubleshooting

| Message | Fix |
|---|---|
| No **Connect GitHub** card | A `GITHUB_APP_*` value is missing, or the backend was not restarted |
| *"The connection request expired"* | Try again; the link is valid for 10 minutes |
| *"GitHub did not return the expected details"* | **Request user authorization (OAuth) during installation** is not ticked, or the Callback URL does not match exactly |
| *"DevPilot is not installed on owner/repo"* | Click **Add or change repositories** and tick that repo |
| *"belongs to a GitHub installation that is not connected"* | The repo is owned by an account you have not connected |
| *"no longer installed"* | Click **Connect GitHub** again |
| Push `403` | The App's **Contents** permission is not **Read and write**; change it on the App page, then accept the new permissions on your installation |
