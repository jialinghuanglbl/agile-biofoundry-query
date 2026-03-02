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
- The GitHub Actions workflow may not work reliably if Streamlit redirects unauthenticated requests to a login page. In that case or if you just prefer a simpler solution, you can use a free external uptime/monitoring service instead.
  - **UptimeRobot** offers a free tier with 5-minute checks. Just create an account, add a "HTTP(s)" monitor pointing at your app URL, and it will periodically load the page like a browser.
  - **Pingdom**, **Cron-job.org**, or similar services also work—any tool that performs a real browser or HTTP GET request will keep the app awake.
- If you switch to an external service you can disable or remove the `.github/workflows/keep_alive.yml` workflow; it exists only as a convenience fallback.
