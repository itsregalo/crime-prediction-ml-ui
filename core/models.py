from django.db import models

class Crime(models.Model):
    id = models.IntegerField(primary_key=True)
    case_number = models.CharField(max_length=20)
    date = models.DateTimeField()
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
    updated_on = models.DateTimeField()
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    location = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return self.case_number
    
    class Meta:
        managed = False
        db_table = 'crime'
