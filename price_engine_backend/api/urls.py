from django.urls import path
from .views import process_files

urlpatterns = [
    path('process/', process_files, name='process_files'),
]