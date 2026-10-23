# Streamlit deployment

Target app: <https://asl-recognition-7.streamlit.app/>.

## Deployment settings

Use the GitHub repository `Vansh-7/sign_lang_recognition_web-app`, branch
`main` after merging, entry point `app.py`, and Python **3.11**. To preview
the pending changes, use branch `fix/streamlit-resource-usage` instead.

Community Cloud runs from the repository root. Dependencies and configuration
therefore remain at the root even though the application code is packaged
under `asl_recognition/`. Model and image paths resolve from the package
location, not the process's working directory.

- `requirements.txt`: pinned Python runtime dependencies, including a Starlette
  version compatible with Streamlit's media response middleware.
- `packages.txt`: native video libraries, including `libgl1`.
- `.streamlit/config.toml`: deployment defaults with file watching disabled.
- `models/` and `assets/`: required runtime files, committed with the code.
- `data/`, `notebooks/`, `training/`: research artifacts; never read by the app.

Do not install `requirements-dev.txt` or `training/requirements.txt` in the
Cloud deployment. TensorFlow and Keras are intentionally absent from the
runtime dependencies. Do not rename `app.py` in the Cloud settings.

## TURN secrets

Open the deployed app's settings and paste the following into **Secrets**,
substituting your own credentials:

```toml
[twilio]
account_sid = "YOUR_TWILIO_ACCOUNT_SID"
auth_token = "YOUR_TWILIO_AUTH_TOKEN"
```

For local development, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml`. Git ignores the latter file. The server exchanges
these credentials for temporary TURN server access; the browser receives
temporary ICE credentials, not the Twilio account credentials. Tokens are
requested for two hours and the server cache refreshes hourly.

STUN is the fallback if TURN credentials are absent or unavailable. That
fallback is insufficient on some mobile, corporate, and restrictive networks.
Twilio relay usage can incur charges; manage quotas and billing in Twilio.

## Update the existing app

1. Review and merge the deployment branch into the branch configured in Cloud.
2. Keep the main file path set to `app.py`.
3. Save the Twilio credentials in Cloud secrets if TURN is needed.
4. Watch the build logs. Dependency changes trigger a rebuild; reboot from
   **Manage app** if the app remains on a failed or stale process.
5. Confirm that installation finishes before testing the browser camera.

If the existing app uses an incompatible Python version, changing Python
requires redeploying the app through Community Cloud's current workflow.
Record the repository, branch, entry point, URL, and secrets before doing so;
see the official Python upgrade guide below.

## Verify after deployment

- The title, camera controls, and reference image load without an exception.
- Camera access works over HTTPS; microphone access is not requested.
- A stable sign appends a letter; edit, clear, download, and speak work.
- Stopping and restarting the camera works without a stale tracker.
- A second browser session has its own text and tracker.
- Test from the networks your users will use, including mobile data if relevant.
- Watch memory and CPU with the expected number of simultaneous sessions.

The model has 28 pose classes and does not model motion sequences. Evaluate
recognition quality with real hands in different lighting; the automated
checks do not establish real-world recognition accuracy.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Cloud shows only “Error running app” | Open Manage app and read the first build/runtime error; the public page hides tracebacks. |
| `libGL.so.1` is missing | Confirm `packages.txt` includes `libgl1` and rebuild. |
| Dependency installation fails | Confirm Python 3.11 and the committed root requirements; avoid mixing training packages into this environment. |
| Model or reference image is missing | Confirm both `models/` and `assets/` were deployed with the package; do not copy only `app.py`. |
| Camera permission denied | Allow the camera for this HTTPS origin and close other camera consumers. |
| Camera stays connecting | Verify Twilio credentials and TURN reachability; test a different network. |
| Recognition is delayed | Check host CPU, concurrent sessions, and network latency; lower the sample count if a shorter hold is desired. |
| Memory exhaustion recurs | Verify TensorFlow is absent from the installed runtime, inspect session load, and restart failed sessions; size hosting for measured concurrency. |
| Speech fails | Confirm outbound access to gTTS and try again; recognition does not depend on speech generation. |

Do not disable TLS checks, CORS, or XSRF protections to solve camera issues.

## Rollback

Revert the problematic commit on the deployed branch, push the revert, and
wait for Cloud to rebuild. Code, assets, model, and requirements must be
rolled back together. Reverting to the original app also restores its old
resource problems. Secrets are managed separately from Git.

## Official documentation

- [Repository layout](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Manage an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [Change Python version](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python)
