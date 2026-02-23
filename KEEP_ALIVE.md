# Keep-Alive / Keep App Awake

This repository includes a GitHub Actions workflow that can periodically ping your deployed Streamlit app to prevent it from idling.

How it works
- The workflow `.github/workflows/keep_alive.yml` runs on a schedule (every 5 minutes) and issues a HEAD/GET request to the app URL.

Setup steps
1. Go to your repository Settings → Secrets → Actions (or `Settings > Secrets and variables > Actions`).
2. Add a repository secret named `STREAMLIT_APP_URL` with the full URL of your Streamlit app (for example `https://agile-biofoundry-query-mwvbcfr5blktuyjddgkddj.streamlit.app/`).
3. Ensure GitHub Actions are enabled for the repository.

Notes & alternatives
- This keep-alive approach pings your app but is not a substitute for a paid "always-on" hosting plan.
- If you prefer a hosted monitoring service, consider UptimeRobot or Pingdom which provide easy UI-based setups.
