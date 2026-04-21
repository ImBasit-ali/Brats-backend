# Backend Files Step-by-Step Guide

## 1) How the backend runs end-to-end

1. Process starts with manage.py (local) or gunicorn via Procfile (deploy).
2. Django loads config.settings and reads environment values from backend/.env.
3. Installed apps and middleware are initialized (DRF, CORS, WhiteNoise, segmentation app).
4. URL routing loads config.urls, then includes segmentation.urls under /api/.
5. Upload requests hit segmentation.views.
6. For POST /api/segment/, uploaded files are validated, normalized to stacked input, job rows are saved, and mock processing thread starts.
7. Mock pipeline in segmentation.tasks updates job progress, runs inference for NIfTI input, saves ET/WT/TC outputs, stores metrics.
8. Frontend polls status/result endpoints and downloads the output file from /api/segment/{id}/download/.

---

## 2) File-by-file explanation

## Project and runtime files

### backend/manage.py
Purpose:
- Django command entrypoint.

Step-by-step:
1. Sets DJANGO_SETTINGS_MODULE to config.settings.
2. Imports execute_from_command_line.
3. Runs Django commands like runserver, migrate, collectstatic.

### backend/Procfile
Purpose:
- Railway process definitions.

Step-by-step:
1. release runs migrations and collectstatic.
2. web starts gunicorn using config.wsgi:application.

### backend/runtime.txt
Purpose:
- Pins Python runtime for deploy.

Step-by-step:
1. Declares python-3.11.9 for platform build/runtime alignment.

### backend/requirements.txt
Purpose:
- Python dependency lock range.

Step-by-step:
1. Installs Django/DRF/CORS.
2. Installs NIfTI/image/scientific stack (nibabel, numpy, Pillow, scipy).
3. Installs serving/deploy tools (gunicorn, whitenoise, dj_database_url, psycopg2-binary).
4. Installs model runtime stack (keras, jax, jaxlib).

### backend/.env
Purpose:
- Environment configuration file (secret values are redacted in this guide).

Detected keys:
- DEBUG
- DATABASE_URL
- ALLOWED_HOSTS
- CORS_ALLOWED_ORIGINS
- CORS_ALLOWED_ORIGIN_REGEXES
- CSRF_TRUSTED_ORIGINS
- MODEL_PATH

Step-by-step:
1. config.settings loads this file using dotenv.
2. Values override defaults for DB, hosts, CORS/CSRF, debug mode, model path.

### backend/.gitignore
Purpose:
- Prevents committing local/generated artifacts.

Step-by-step:
1. Ignores virtual envs, pycache, db.sqlite3, media/static outputs, .env files.
2. Keeps repository clean for deployment.

### backend/MODEL_PATH_SETUP.md
Purpose:
- Human-readable setup guide for model path configuration.

Step-by-step:
1. Explains where model file is expected.
2. Explains MODEL_PATH usage across local and Railway.
3. Explains restart and verification process.

### backend/db.sqlite3
Purpose:
- Local development database.

Step-by-step:
1. Stores segmentation jobs and uploaded file rows when using sqlite locally.
2. Not used in Railway production when DATABASE_URL points to Postgres.

### backend/model_assets/brats_3d_unet_final.keras
Purpose:
- Trained Keras model artifact.

Step-by-step:
1. Loaded lazily by segmentation.model_loader.get_model.
2. Used by segmentation.inference to produce ET/WT/TC masks.

### backend/model_assets/brats-model.log
Purpose:
- CSV-style training/validation metric log.

Step-by-step:
1. Stores epoch metrics from training.
2. Used for audit/reference, not required at request runtime.

---

## Django project config package

### backend/config/__init__.py
Purpose:
- Marks config directory as Python package.

Step-by-step:
1. Enables imports like config.settings.

### backend/config/settings.py
Purpose:
- Central Django configuration.

Step-by-step:
1. Reads .env values (booleans/csv helpers).
2. Configures ALLOWED_HOSTS, CORS, CSRF.
3. Selects DB URL (local sqlite or Railway/Postgres).
4. Registers apps/middleware.
5. Configures static/media storage.
6. For Railway: sets proxy headers and uses /tmp-based media/upload temp directories.
7. Enables multipart/form parsers.
8. Sets large upload limits and upload handlers.

### backend/config/urls.py
Purpose:
- Root URL map.

Step-by-step:
1. Adds /admin/.
2. Adds /api/ routes from segmentation.urls.
3. Serves /media/ files using document_root = MEDIA_ROOT.

### backend/config/wsgi.py
Purpose:
- WSGI app entry for gunicorn.

Step-by-step:
1. Sets DJANGO_SETTINGS_MODULE.
2. Exposes application callable.

---

## Segmentation app files

### backend/segmentation/__init__.py
Purpose:
- App package marker and default app config reference.

Step-by-step:
1. Points to SegmentationConfig.

### backend/segmentation/apps.py
Purpose:
- Django app config metadata.

Step-by-step:
1. Names app segmentation.
2. Sets verbose display name.

### backend/segmentation/models.py
Purpose:
- Database schema for jobs and files.

Step-by-step:
1. Defines SegmentationJob with status/progress/metrics/result fields.
2. Defines UploadedFile linked to a job with modality + file storage path.
3. Uses UUID primary keys.

### backend/segmentation/admin.py
Purpose:
- Admin interface customization.

Step-by-step:
1. Registers SegmentationJob and UploadedFile in admin.
2. Adds inline uploaded-file view under each job.
3. Adds filters/list/search/read-only fields.

### backend/segmentation/migrations/__init__.py
Purpose:
- Migration package marker (empty file).

Step-by-step:
1. Allows Django migration discovery/import.

### backend/segmentation/migrations/0001_initial.py
Purpose:
- Initial DB migration.

Step-by-step:
1. Creates SegmentationJob table.
2. Creates UploadedFile table with FK to SegmentationJob.
3. Applies field defaults, choices, ordering.

### backend/segmentation/stacking.py
Purpose:
- Validation and stacking utilities for NIfTI/PNG inputs.

Step-by-step:
1. Infers extension (.nii/.nii.gz/.png).
2. Validates upload combination:
   - exactly 1 stacked-like file OR exactly 4 modalities
   - rejects 2/3 modality partial uploads
   - rejects duplicates and wrong modality sets
3. For NIfTI: loads 3D volumes, checks same dimensions, stacks to 4D channels.
4. For PNG: loads grayscale per modality and merges RGBA channels.

### backend/segmentation/model_loader.py
Purpose:
- Thread-safe lazy model loading.

Step-by-step:
1. Resolves model path from MODEL_PATH or default path.
2. Lazily imports keras and ops only when needed.
3. Uses global lock to avoid duplicate parallel loads.
4. Loads model with custom instance_normalization function.

### backend/segmentation/inference.py
Purpose:
- Converts stacked NIfTI into ET/WT/TC masks using the model.

Step-by-step:
1. Loads stacked input with nibabel.
2. Normalizes and reshapes to model input format.
3. Runs prediction.
4. Normalizes prediction layout to channels-last.
5. Splits channels into ET/WT/TC (with fallback for 1/2 channel outputs).
6. Restores prediction to original spatial shape.
7. Applies anatomical foreground mask to reduce background artifacts.
8. Returns masks + affine/header for output save.

### backend/segmentation/tasks.py
Purpose:
- Background mock segmentation pipeline.

Step-by-step:
1. Starts daemon thread per job.
2. Updates step name/progress with simulated delays.
3. Creates outputs:
   - For NIfTI: runs real inference pipeline and saves ET/WT/TC NIfTI masks.
   - For PNG: creates threshold-based mock masks.
4. Writes output files as UploadedFile records.
5. Sets job.segmentation_file to whole tumor mask.
6. Generates mock metrics and marks job done, or stores error.

### backend/segmentation/serializers.py
Purpose:
- API response shaping for status/result endpoints.

Step-by-step:
1. Serializes UploadedFile metadata.
2. Builds structured progress object.
3. Builds result payload with:
   - download_url
   - model_input_url
   - overlays map
4. Converts media paths to absolute URLs and forces https on Railway.

### backend/segmentation/urls.py
Purpose:
- Segmentation endpoint map.

Step-by-step:
1. POST /stack/ (legacy alias).
2. POST /segment/ create job.
3. POST /segment/stack/ stack preview only.
4. GET /segment/{id}/status/.
5. GET /segment/{id}/result/.
6. GET /segment/{id}/download/.

### backend/segmentation/views.py
Purpose:
- Request handlers for upload, stacking preview, polling, result, and download.

Step-by-step:
1. create_segmentation:
   - validates upload exists
   - warms model load
   - creates job
   - stores raw uploads
   - validates modality combination
   - creates stacked input (from 4 modalities or duplicates single into 4 channels)
   - starts background mock pipeline
   - returns job id/status
2. stack_preview:
   - accepts files list or named modality fields
   - picks one preview source safely
   - saves preview file to storage under previews/
   - returns public preview_url
3. get_segmentation_status:
   - returns current job status/progress and error text when applicable
4. get_segmentation_result:
   - returns 202 while processing
   - returns full serializer result when done/error
5. download_segmentation:
   - streams final segmentation file as attachment when done

---

## 3) Generated and metadata files under backend

These files/folders exist but are not core business logic:

- backend/.venv/...: local virtual environment packages and binaries.
- backend/media/...: uploaded and generated runtime files.
- backend/staticfiles/...: collectstatic output.
- backend/config/__pycache__/*.pyc and backend/segmentation/__pycache__/*.pyc: Python bytecode cache.
- backend/.git/... and backend/.git/hooks/*.sample: repository metadata and hook templates (appears as nested git metadata inside backend).

Step-by-step role:
1. They support runtime/build tooling.
2. They should not be edited for feature logic.
3. They are generally regenerated automatically.

---

## 4) Request flow map (quick reference)

1. Client uploads files -> POST /api/segment/stack/ for preview.
2. Client uploads files -> POST /api/segment/ to create processing job.
3. Backend writes UploadedFile rows and normalized stacked input.
4. Background thread runs pipeline and writes output masks.
5. Client polls /status then fetches /result.
6. Client downloads mask via /download.
