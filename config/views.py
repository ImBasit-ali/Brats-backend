from django.http import JsonResponse

def worker_health_check(request):
    return JsonResponse({"status": "ok"})