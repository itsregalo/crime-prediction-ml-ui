from import_export import resources
from .models import Crime, ProcessedCrimeData

class CrimeResource(resources.ModelResource):
    class Meta:
        model = Crime


class ProcessedCrimeDataResource(resources.ModelResource):
    class Meta:
        model = ProcessedCrimeData