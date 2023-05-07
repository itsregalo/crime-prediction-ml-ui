from django.urls import path
from .views import *

app_name = 'core'

urlpatterns = [
    # ... your other URL patterns ...
    path('', IndexView, name='index'),
    path('data/', DataTableView, name='data'),
    path('data-description', data_description, name='data-description'),
    path('preprocess-data/', clean_data, name='preprocess-data'),
    path('processed-data/', processed_data, name='processed_data'),
    path('import-data/', import_data, name='import-data'),
    path('data-analytics/', data_analytics, name='data-analytics'),
    path('train-model/', train_model, name='train-model'),
    path('nerd-stats/', nerd_statistics, name='nerd-stats'),
]
