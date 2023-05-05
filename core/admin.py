
from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from .models import Crime, ProcessedCrimeData, latest_model_statistics, crime_type_model_statistics, latest_predictions_plots

class CrimeAdmin(ImportExportModelAdmin):
    list_display = ('case_number', 'date', 'block', 'primary_type', 'description', 'location_description', 'arrest', 'domestic', 'beat', 'district',
                    'ward', 'community_area', 'fbi_code', 'x_coordinate', 'y_coordinate', 'year', 'updated_on', 'latitude', 'longitude', 'location')
    list_filter = ('date', 'primary_type', 'description', 'location_description', 'arrest', 'domestic', 'beat',
                   'district', 'ward', 'community_area', 'fbi_code', 'year', 'updated_on', 'latitude', 'longitude', 'location')
    search_fields = ('case_number', 'block', 'primary_type', 'description', 'location_description', 'arrest', 'domestic', 'beat', 'district',
                     'ward', 'community_area', 'fbi_code', 'x_coordinate', 'y_coordinate', 'year', 'updated_on', 'latitude', 'longitude', 'location')
    ordering = ('date',)
    list_per_page = 100

admin.site.register(Crime, CrimeAdmin)

class ProcessedCrimeDataAdmin(ImportExportModelAdmin):
    list_display = ('date', 'primary_type', 'description', 'location_description', 'arrest', 'domestic', 'beat', 'district', 'ward', 'community_area', 'fbi_code', 'x_coordinate', 'y_coordinate', 'year', 'updated_on', 'latitude', 'longitude', 'day_of_week', 'month', 'time', 'primary_type_grouped', 'zone', 'season', 'loc_grouped')
    list_filter = ('date', 'primary_type', 'location_description', 'arrest', 'domestic', 'beat', 'district', 'ward', 'community_area', 'fbi_code', 'year', 'day_of_week', 'month', 'time', 'primary_type_grouped', 'zone', 'season', 'loc_grouped')
    search_fields = ('date', 'primary_type', 'location_description', 'arrest', 'domestic', 'beat', 'district', 'ward', 'community_area', 'fbi_code', 'year', 'day_of_week', 'month', 'time', 'primary_type_grouped', 'zone', 'season', 'loc_grouped')

admin.site.register(ProcessedCrimeData, ProcessedCrimeDataAdmin)


class LatestModelStatisticsAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'model_accuracy', 'model_precision', 'model_recall', 'model_error', 'model_f1_score', 'model_confusion_matrix', 'model_classification_report', 'timestamp')
    list_filter = ('model_name', 'model_accuracy', 'model_precision', 'model_recall', 'model_error', 'model_f1_score', 'model_confusion_matrix', 'model_classification_report', 'timestamp')
    search_fields = ('model_name', 'model_accuracy', 'model_precision', 'model_recall', 'model_error', 'model_f1_score', 'model_confusion_matrix', 'model_classification_report', 'timestamp')
    ordering = ('-timestamp',)
    filter_horizontal = ()
    list_filter = ()
    fieldsets = ()

class CrimeTypeModelStatisticsAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'model_accuracy', 'model_precision', 'model_recall', 'model_error', 'model_f1_score', 'model_confusion_matrix', 'model_classification_report', 'timestamp')
    list_filter = ('model_name', 'model_accuracy', 'model_precision', 'model_recall', 'model_error', 'model_f1_score', 'model_confusion_matrix', 'model_classification_report', 'timestamp')
    search_fields = ('model_name', 'model_accuracy', 'model_precision', 'model_recall', 'model_error', 'model_f1_score', 'model_confusion_matrix', 'model_classification_report', 'timestamp')
    ordering = ('-timestamp',)
    filter_horizontal = ()
    list_filter = ()
    fieldsets = ()

admin.site.register(latest_model_statistics, LatestModelStatisticsAdmin)
admin.site.register(crime_type_model_statistics, CrimeTypeModelStatisticsAdmin)
admin.site.register(latest_predictions_plots)
