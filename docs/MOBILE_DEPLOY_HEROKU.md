# Deploy Kronos Market Foresight from your phone
# (GitHub Student Pack → Heroku)

**Owner:** Vansh Bhasin · Repo: `bhasinvansh05/Kronos`  
**Goal:** Public HTTPS URL, mostly using only your phone’s browser.

Use **Safari** (iPhone) or **Chrome** (Android). Turn on **Request Desktop Website** when a page looks broken — Heroku’s dashboard is easier in desktop mode.

Estimated time: **45–90 minutes** the first time (account linking + first build).

---

## What you will end up with

- A public site like `https://YOUR-APP-NAME.herokuapp.com`
- Powered by your Student Pack **Heroku $13/mo credit**
- Running the `webapp/` Flask app with **Kronos-mini** (smallest / cheapest)

---

## Before you start (checklist)

1. You are a **verified GitHub Student** (Student Developer Pack active).
2. You can open [https://education.github.com/pack](https://education.github.com/pack) while logged into GitHub.
3. Your code is on GitHub: [https://github.com/bhasinvansh05/Kronos](https://github.com/bhasinvansh05/Kronos)  
   Prefer the branch that has the webapp (e.g. `cursor/kronos-prediction-webapp-78e3` or `master` after merge).
4. Phone charger nearby — first Docker build can take **10–20+ minutes**.

---

# PART A — Redeem Heroku from the Student Pack (phone)

### A1. Open the pack
1. On your phone, open the browser.
2. Go to: **https://education.github.com/pack**
3. Tap **Sign in** if needed → use your GitHub account (`bhasinvansh05`).
4. Confirm you see offers (you should see **Heroku**).

### A2. Find Heroku
1. On the pack page, use the search box if available → type **Heroku**.
2. Or scroll categories until you see **Heroku**.
3. Tap the **Heroku** offer card.

### A3. Claim the offer
1. Tap **Get offer** / **Access offer** / **Sign up** (wording varies).
2. You will be sent to Heroku’s student page or signup.
3. Create a Heroku account with the **same email** as your GitHub/student email if possible.
4. If Heroku asks to verify email → open your mail app → tap the link.
5. Finish any “Welcome / Create team / Accept terms” screens.

### A4. Confirm credits
1. Open **https://dashboard.heroku.com**
2. Tap your **avatar / account menu** (often top-right; in mobile menu top-left ☰).
3. Open **Account settings** → look for **Billing**.
4. You should see student / platform credits (~**$13/month**).  
   If not visible yet, wait a few minutes, or reopen the pack offer and ensure it says **Redeemed**.

> If Heroku says the offer is already used / unavailable: skip to **PART F (Azure backup)** or use a free Hugging Face Space later.

---

# PART B — Create the Heroku app (phone)

### B1. New app
1. Open **https://dashboard.heroku.com/new-app**  
   (or Dashboard → **New** → **Create new app**).
2. **App name:** something unique, e.g. `vansh-kronos` or `kronos-foresight-vansh`  
   (this becomes `https://THAT-NAME.herokuapp.com`).
3. **Region:** United States (or Europe — either is fine).
4. Tap **Create app**.

### B2. Switch to Container stack (required for PyTorch)
PyTorch is too large for a normal Heroku “buildpack” slug. This repo includes Docker files.

1. On the app page, open the **Settings** tab.
2. Find **Stack**.
3. If you see stack options / “Container”, select **container**.  
   If you only have Heroku-22/24: we’ll set it via GitHub deploy using `heroku.yml` (already in the repo). Keep going.

### B3. Add a stronger dyno type (important)
Torch needs RAM. Eco (512MB) often crashes.

1. Open the **Resources** tab.
2. Under **Dyno formation** / **Free/Eco/Basic**, change the web dyno to the highest tier your **$13 credit** can cover for a month.  
   Prefer **Basic** (or Standard if credit allows) over Eco.
3. Make sure the **web** dyno is **ON** (not scaled to 0).

> If money/credit is tight: stay on the cheapest dyno but **only** use Kronos-mini (this guide assumes mini).

---

# PART C — Connect GitHub and deploy (phone)

### C1. Open Deploy tab
1. In your Heroku app, tap **Deploy**.
2. Under **Deployment method**, choose **GitHub**.
3. Tap **Connect to GitHub** → authorize Heroku to see your repos.  
   - Sign in to GitHub if prompted.  
   - Tap **Authorize Heroku**.

### C2. Link this repository
1. Search for **`Kronos`** (or `bhasinvansh05/Kronos`).
2. Tap **Connect** next to **bhasinvansh05/Kronos**.

### C3. Pick the branch
1. Under **Automatic deploys** / **Manual deploy**, choose the branch that contains the webapp:
   - Prefer: `cursor/kronos-prediction-webapp-78e3` (if not merged yet), **or**
   - `master` / `main` (after you merge the PR).
2. Optional but recommended: enable **Automatic deploys** → **Enable Automatic Deploys**.

### C4. First deploy
1. Scroll to **Manual deploy**.
2. Tap **Deploy Branch**.
3. Keep the phone awake / screen on.  
   First build can take a long time (Docker + torch + model).
4. Wait until you see **“Your app was successfully deployed.”**  
   If it fails, jump to **PART E — Fixes**.

### C5. Open the site
1. Tap **Open app** (top right), or visit:  
   `https://YOUR-APP-NAME.herokuapp.com`
2. You should see **KRONOS · Market foresight · by Vansh Bhasin**.
3. Search **AAPL** → wait for chart → set **24h** → **Predict Next Hours**.

---

# PART D — Config vars on phone (do this if predict fails)

1. Heroku app → **Settings** → **Reveal Config Vars**.
2. Add / confirm:

| Key | Value | Why |
|-----|-------|-----|
| `MODEL_KEY` | `kronos-mini` | Keeps memory low |
| `WEB_CONCURRENCY` | `1` | One worker only |
| `HF_HOME` | `/tmp/huggingface` | Cache location |

3. Tap **Open app** again after saving.
4. Resources → restart dynos if there’s a **Restart all dynos** option.

---

# PART E — Common failures & exact fixes

### E1. Build failed: out of space / slug too large
- You must deploy via **Docker / container** (`Dockerfile` + `heroku.yml` in repo).
- Re-check Deploy method isn’t a classic Python buildpack-only deploy.
- Redeploy branch after confirming those files exist on GitHub.

### E2. App crashes / R14 memory errors (in Logs)
1. App → **More** → **View logs** (or **Activity**).
2. If you see memory quota exceeded:
   - Upgrade dyno (Resources).
   - Confirm `MODEL_KEY=kronos-mini` and `WEB_CONCURRENCY=1`.
   - Don’t select Kronos-base in the UI on a tiny dyno.

### E3. “Application error” but build succeeded
1. View logs.
2. Look for missing module / port errors.
3. Confirm the dyno is **up** (Resources → web dyno ON).
4. Hard-refresh the browser (or clear site data).

### E4. Model download timeout on first predict
The Docker image is set up to preload **Kronos-mini**. If logs show Hugging Face download failures:
- Redeploy (network blip).
- Or retry Predict after 1–2 minutes.

### E5. GitHub branch doesn’t have webapp
Merge PR **#3** into `master` on GitHub (phone):
1. Open https://github.com/bhasinvansh05/Kronos/pull/3  
2. Tap **Merge pull request** → **Confirm merge**.  
3. In Heroku Deploy, select `master` → **Deploy Branch**.

---

# PART F — Optional: custom domain from Student Pack (.me)

1. On https://education.github.com/pack find **Namecheap** / **.me domain** offer.
2. Redeem → register something like `vanshbhasin.me` (or similar available name).
3. In Heroku → Settings → **Domains** → Add domain `kronos.YOURNAME.me` (or apex).
4. Heroku shows a DNS target. In Namecheap DNS (phone browser):
   - Add **CNAME** `kronos` → `YOUR-APP-NAME.herokuapp.com`  
   (exact target shown by Heroku).
5. Wait 5–60 minutes for DNS. Open `https://kronos.YOURNAME.me`.

---

# PART G — Backup if Heroku won’t fit (Azure for Students)

Do this only if Heroku keeps OOMing or builds fail.

1. Student Pack → redeem **Microsoft Azure** (Azure for Students).
2. Open **https://portal.azure.com** on your phone (desktop site mode).
3. Create **Web App** / **Container Apps** (Linux).
4. Deployment Center → GitHub → pick `bhasinvansh05/Kronos` → Dockerfile.
5. Set app setting `PORT` / listen port as Azure instructs.
6. Browse the `*.azurewebsites.net` URL.

Azure’s student credit usually allows a bit more RAM headroom than Heroku Eco.

---

# PART H — Day-to-day from your phone

| Task | Where |
|------|--------|
| See site | `https://YOUR-APP-NAME.herokuapp.com` |
| Push updates | Merge to the connected branch on GitHub → auto-deploy |
| Logs | Heroku → More → View logs |
| Turn off to save credits | Resources → web dyno → scale to **0** (site goes offline) |
| Turn back on | Scale web dyno to **1** |

---

## Security / product notes

- Add a clear disclaimer on the live site (already in the UI): **not financial advice**.
- Don’t put secrets in the repo; use Heroku **Config Vars**.
- Public demos should default to **Kronos-mini** and 24h forecasts.

---

## Quick success test

1. Open your Heroku URL.  
2. Search `AAPL`.  
3. Range `1M` loads candles.  
4. Predict 24h with Kronos-mini finishes without “Application error”.  
5. Footer shows **Vansh Bhasin**.

You’re live.
