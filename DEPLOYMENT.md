# Streamlit Cloud Deployment Guide

## Prerequisites
✅ Your app is already prepared! The following are ready:
- GitHub repository: `https://github.com/darkmatter11235/sunworks_hybrid_optimizer.git`
- `requirements.txt` with necessary dependencies
- `packages.txt` for system dependencies
- Demo data CSV files included in repository
- `.streamlit/config.toml` for app configuration

## Step-by-Step Deployment

### 1. Sign in to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your GitHub account

### 2. Deploy Your App
1. Click **"New app"** (or "Create app") button
2. Fill in the deployment form:
   - **Repository**: `darkmatter11235/sunworks_hybrid_optimizer`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (optional): Choose a custom URL or use the auto-generated one

3. Click **"Deploy!"**

### 3. Wait for Deployment
- Streamlit Cloud will:
  - Clone your repository
  - Install dependencies from `requirements.txt`
  - Install system packages from `packages.txt` (if needed)
  - Start your app

- Initial deployment takes 2-5 minutes
- You'll see a build log showing the progress

### 4. Your App is Live! 🎉
Once deployed, you'll get a URL like:
- `https://[your-app-name].streamlit.app`

## App Features Available on Deployment

Users can:
- ✅ **Use Demo Data** - Pre-loaded 830 MW solar + 50 MW BESS demo configuration
- ✅ **Upload Excel Files** - Upload their own hybrid RE system models
- ✅ **Run Simulations** - Hourly energy flow simulation with battery dispatch
- ✅ **Calculate LCOE** - Levelized cost of energy analysis
- ✅ **Optimize Systems** - Find optimal solar/wind/BESS configurations

## Managing Your App

### Update Your App
Any push to the `main` branch will automatically redeploy your app:
```bash
git add .
git commit -m "Update description"
git push origin main
```

### View Logs
- Click on your app in Streamlit Cloud dashboard
- Click "Manage app" → "Logs" to see runtime logs
- Useful for debugging if something goes wrong

### App Settings
In the Streamlit Cloud dashboard, you can:
- View analytics (visitors, page views)
- Configure secrets (if needed for API keys)
- Reboot or delete the app
- Change app visibility (public/private)

## Troubleshooting

### App won't start?
- Check the build logs for errors
- Common issues:
  - Missing dependencies in `requirements.txt`
  - Import errors in Python code
  - Path issues (use relative paths)

### App is slow?
- Streamlit Cloud free tier has resource limits
- Optimize data loading with `@st.cache_data`
- Consider upgrading to paid tier for better performance

### Demo data not working?
- Verify CSV files are in the repository: `standalone_data/generation_profiles.csv` and `standalone_data/load_profile.csv`
- Check that they're not in `.gitignore`

## Resources

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **Your Repository**: [github.com/darkmatter11235/sunworks_hybrid_optimizer](https://github.com/darkmatter11235/sunworks_hybrid_optimizer)

## What Was Changed for Deployment

The following files were updated to prepare for Streamlit Cloud:

1. **`.gitignore`** - Modified to include demo data CSV files
2. **`requirements.txt`** - Removed development dependencies (pyinstaller, pytest)
3. **`standalone_data/`** - Added generation_profiles.csv and load_profile.csv to git

These changes ensure the demo mode works on Streamlit Cloud without requiring users to upload data.

---

**Need help?** Check the Streamlit Community forum or open an issue in your GitHub repository.
