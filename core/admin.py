
from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from .models import Crime, ProcessedCrimeData

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
