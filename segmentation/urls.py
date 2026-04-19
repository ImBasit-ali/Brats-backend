from django.urls import path
from . import views

urlpatterns = [
    path('stack/', views.stack_preview, name='stack-preview-legacy'),
    path('segment/', views.create_segmentation, name='create-segmentation'),
    path('segment/stack/', views.stack_preview, name='stack-preview'),
    path('segment/<uuid:job_id>/status/', views.get_segmentation_status, name='segmentation-status'),
    path('segment/<uuid:job_id>/result/', views.get_segmentation_result, name='segmentation-result'),
    path('segment/<uuid:job_id>/download/', views.download_segmentation, name='segmentation-download'),
]
