# Model Path Setup

This project loads the segmentation model from a file path. Use this guide to point Django to the correct model file.

## 1. Model file location

The model file is already included in the repository at:

`backend/model_assets/brats_3d_unet_final.keras`

If you want to use a different model file, place it anywhere on disk and use that full path instead.

## 2. Environment variable name

Set this environment variable:

`MODEL_PATH`

This is the path that `backend/segmentation/model_loader.py` should use when loading the model.

## 3. Example `.env` entry

Add this to your backend `.env` file:

```env
MODEL_PATH=backend/model_assets/brats_3d_unet_final.keras
```

If you are setting an absolute Windows path, use:

```env
MODEL_PATH=C:\Users\basit\OneDrive\Desktop\brats-UI\backend\model_assets\brats_3d_unet_final.keras
```

## 4. How Django reads it

The backend loads the model from `MODEL_PATH` first. If that variable is missing, the app may try another fallback depending on the current code version.

To make the behavior predictable in local development and deploy, always set `MODEL_PATH` explicitly.

## 5. Where to set it

Set `MODEL_PATH` in the place that matches your environment:

- Local development: `backend/.env`
- Railway backend service: Railway environment variables
- Docker or server deploy: the container or server environment

## 6. Restart after changing it

After editing the environment variable, restart the backend server so Django picks up the new value.

For local development:

```bash
cd backend
python manage.py runserver
```

If the server was already running, stop it and start it again.

## 7. Verify the model loads

After restart, watch the backend logs.

You want to see that the model is loaded without errors, and you should not see this message:

`MODEL_URL environment variable is not set`

If you do, the backend is still using the wrong loader path or the environment variable was not applied.

## 8. If you want to use a download URL instead

If the model is not stored locally, you can still use a download URL by setting:

```env
MODEL_URL=https://example.com/brats_3d_unet_final.keras
```

In that case, `MODEL_PATH` should still point to the local file location where the downloaded model will be saved.

## 9. Recommended setup

Use this combination for a stable deploy:

```env
MODEL_PATH=backend/model_assets/brats_3d_unet_final.keras
```

That keeps the model file local to the project and avoids dependency on a remote download during startup.
