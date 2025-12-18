# Deploying AlphaSAGE Dashboard to Vercel

## 1. Push to GitHub
First, create a new repository on GitHub (e.g., `alpha-sage-dashboard`) and push this code:

```bash
cd alpha-sage-dashboard
git add .
git commit -m "Initial commit"
# Replace with your repo URL
git remote add origin https://github.com/YOUR_USERNAME/alpha-sage-dashboard.git
git push -u origin main
```

## 2. Import to Vercel
1.  Go to [Vercel Dashboard](https://vercel.com/dashboard).
2.  Click **Add New...** -> **Project**.
3.  Select your GitHub repository (`alpha-sage-dashboard`).
4.  Vercel will auto-detect the Next.js framework.
5.  Click **Deploy**.

## 3. Verify
Once deployed, Vercel will give you a live URL (e.g., `alpha-sage-dashboard.vercel.app`).
Visit the URL to see your interactive dashboard.

## Note on Data
The CSV files in `public/data` are static. To update the data:
1.  Run `run_adaptive_combination.py` locally.
2.  Copy the new CSV files to `public/data`.
3.  Commit and push changes to GitHub.
4.  Vercel will automatically redeploy the new data.
