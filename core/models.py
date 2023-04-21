from django.db import models

class Crime(models.Model):
    id = models.IntegerField(primary_key=True, unique=True)
    case_number = models.CharField(max_length=20)
    date = models.CharField(max_length=50)
    block = models.CharField(max_length=50)
    iucr = models.CharField(max_length=10)
    primary_type = models.CharField(max_length=50)
    description = models.CharField(max_length=200)
    location_description = models.CharField(max_length=100)
    arrest = models.BooleanField()
    domestic = models.BooleanField()
    beat = models.IntegerField()
    district = models.IntegerField()
    ward = models.FloatField(null=True, blank=True)
    community_area = models.IntegerField()
    fbi_code = models.CharField(max_length=10)
    x_coordinate = models.FloatField(null=True, blank=True)
    y_coordinate = models.FloatField(null=True, blank=True)
    year = models.IntegerField()
    updated_on = models.CharField(max_length=254)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    location = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return self.case_number
    
    class Meta:
        db_table = 'crime_records'
        verbose_name_plural = 'Crime Records'
        ordering = ('date',)

    def total_crimes(self):
        return self.objects.count()
