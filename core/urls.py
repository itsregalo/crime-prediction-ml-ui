from django.urls import path
from .views import import_data

app_name = 'core'

urlpatterns = [
    # ... your other URL patterns ...
    path('import-data/', import_data, name='import-data'),
]
