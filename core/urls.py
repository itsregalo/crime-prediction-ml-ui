from django.urls import path
from .views import *

app_name = 'core'

urlpatterns = [
    # ... your other URL patterns ...
    path('', IndexView, name='index'),
    path('data/', DataTableView, name='data'),
    path('data-description', data_description, name='data-description'),
    path('preprocess-data/', clean_data, name='preprocess-data'),
    path('import-data/', import_data, name='import-data')
]
