# Deployment Guide

This guide will help you host your **Engine Rating Prediction** project for free using **Render** (for the backend) and **Vercel** (for the frontend).

## Prerequisites
- A GitHub account.
- Accounts on [Render](https://render.com) and [Vercel](https://vercel.com).
- Your code pushed to GitHub.

---

## Part 1: Deploy Backend (Render)

1.  **Log in to Render** and click generic "New +", then select **Web Service**.
2.  Connect your GitHub repository (`Engine-Rating-Prediction`).
3.  Configure the service with the following settings:
    -   **Name**: `engine-rating-backend` (or similar)
    -   **Region**: Closest to you (e.g., Singapore, Frankfurt)
    -   **Root Directory**: `backend` (Important!)
    -   **Runtime**: **Python 3**
    -   **Build Command**: `pip install -r requirements.txt`
    -   **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
    -   **Free Instance**: Select the free tier.
4.  Click **Create Web Service**.
5.  Wait for the deployment to finish. You will see a "Live" status.
6.  **Copy the URL** provided by Render (e.g., `https://engine-rating-backend.onrender.com`). You will need this for the frontend.

> **Note:** The backend might take a minute to wake up on the free tier.

---

## Part 2: Deploy Frontend (Vercel)

1.  **Log in to Vercel** and click **Add New...** > **Project**.
2.  Import your GitHub repository (`Engine-Rating-Prediction`).
3.  Configure the project:
    -   **Framework Preset**: Vite (should be auto-detected).
    -   **Root Directory**: Click "Edit" and select `frontend`.
4.  **Environment Variables**:
    -   Expand the "Environment Variables" section.
    -   **Key**: `VITE_API_URL`
    -   **Value**: The Render URL you copied earlier **(IMPORTANT: Remove the trailing slash `/` if present)**.
        -   Example: `https://engine-rating-backend.onrender.com`
5.  Click **Deploy**.
6.  Wait for the build to complete. Vercel will give you a live URL for your website.

---

## Troubleshooting

-   **Backend 500 Error**: Check the Render logs. It might be an issue with loading the `model.pickle`. Ensure the model file is successfully uploaded to GitHub.
-   **Frontend "Prediction Failed"**:
    -   Check the browser console (F12) for errors.
    -   Ensure the `VITE_API_URL` variable in Vercel is correct and does *not* have a trailing slash.
    -   Wake up the backend by visiting its URL directly in a browser tab first.
