from django.urls import path
from .views import *

app_name = 'core'

urlpatterns = [
    # ... your other URL patterns ...
    path('', IndexView, name='index'),
    path('data/', DataTableView, name='data'),
    path('import-data/', import_data, name='import-data')
]
