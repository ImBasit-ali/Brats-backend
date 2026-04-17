import os

from rest_framework import serializers
from .models import SegmentationJob, UploadedFile


class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedFile
        fields = ['id', 'original_name', 'modality', 'uploaded_at']


class SegmentationJobCreateSerializer(serializers.Serializer):
    """Serializer for creating a new segmentation job."""
    grade = serializers.CharField(default='HGG')
    regions = serializers.CharField(default='{}')
    opacity = serializers.IntegerField(default=70)


class SegmentationJobStatusSerializer(serializers.ModelSerializer):
    """Serializer for job status responses."""
    progress = serializers.SerializerMethodField()

    class Meta:
        model = SegmentationJob
        fields = ['id', 'status', 'progress', 'created_at', 'updated_at']

    def get_progress(self, obj):
        steps = ['Preprocess', 'Inference', 'Postprocess', 'Done']
        return {
            'step': obj.current_step,
            'step_name': obj.current_step_name,
            'total_steps': obj.total_steps,
            'steps': steps,
        }


class SegmentationJobResultSerializer(serializers.ModelSerializer):
    """Serializer for job results."""
    files = UploadedFileSerializer(many=True, read_only=True)
    download_url = serializers.SerializerMethodField()
    model_input_url = serializers.SerializerMethodField()
    overlays = serializers.SerializerMethodField()

    class Meta:
        model = SegmentationJob
        fields = [
            'id', 'status', 'grade', 'regions', 'metrics',
            'segmentation_file', 'files', 'download_url',
            'model_input_url', 'overlays',
            'created_at', 'completed_at',
        ]

    def get_download_url(self, obj):
        if obj.segmentation_file:
            return self._build_absolute_uri(obj.segmentation_file.url)
        return None

    def _build_absolute_uri(self, path):
        request = self.context.get('request')
        if not request:
            return path

        url = request.build_absolute_uri(path)
        if os.environ.get('RAILWAY_ENVIRONMENT') and url.startswith('http://'):
            return f"https://{url[len('http://'):]}"
        return url

    def _build_file_url(self, file_obj):
        if file_obj:
            return self._build_absolute_uri(file_obj.file.url)
        return None

    def get_model_input_url(self, obj):
        model_input = obj.files.filter(modality='stacked').order_by('-uploaded_at').first()
        return self._build_file_url(model_input)

    def get_overlays(self, obj):
        et = obj.files.filter(modality='et_mask').order_by('-uploaded_at').first()
        wt = obj.files.filter(modality='wt_mask').order_by('-uploaded_at').first()
        tc = obj.files.filter(modality='tc_mask').order_by('-uploaded_at').first()
        return {
            'enhancing_tumor': self._build_file_url(et),
            'whole_tumor': self._build_file_url(wt),
            'tumor_core': self._build_file_url(tc),
        }
