from import_export import resources
from .models import Crime

class CrimeResource(resources.ModelResource):
    class Meta:
        model = Crime
