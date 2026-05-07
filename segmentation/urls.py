from django.urls import path
from . import views
from .views import worker_health_check

urlpatterns = [
    path("health/", worker_health_check),
    # path('health/', views.worker_health_check, name='worker-health'),
    path('stack/', views.stack_preview, name='stack-preview-legacy'),
    path('segment/', views.create_segmentation, name='create-segmentation'),
    path('segment/stack/', views.stack_preview, name='stack-preview'),
    path('segment/view-uploads/', views.view_individual_uploads, name='view-individual-uploads'),
    path('segment/<uuid:job_id>/status/', views.get_segmentation_status, name='segmentation-status'),
    path('segment/<uuid:job_id>/result/', views.get_segmentation_result, name='segmentation-result'),
    path('segment/<uuid:job_id>/download/', views.download_segmentation, name='segmentation-download'),
]
