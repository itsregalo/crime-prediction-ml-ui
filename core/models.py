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

"""
Data Types:
date                    datetime64[ns]
block                           object
iucr                            object
primary_type                    object
description                     object
location_description            object
arrest                           int64
domestic                         int64
beat                             int64
district                         int64
ward                           float64
community_area                   int64
fbi_code                        object
x_coordinate                   float64
y_coordinate                   float64
year                             int64
updated_on                      object
latitude                       float64
longitude                      float64
day_of_week                     object
month                           object
time                             int64
primary_type_grouped            object
zone                            object
season                          object
loc_grouped                     object
"""

# model with above data types
class ProcessedCrimeData(models.Model):
    date = models.DateTimeField()
    block = models.CharField(max_length=50)
    iucr = models.CharField(max_length=10)
    primary_type = models.CharField(max_length=50)
    description = models.CharField(max_length=200)
    location_description = models.CharField(max_length=100)
    arrest = models.IntegerField()
    domestic = models.IntegerField()
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
    day_of_week = models.CharField(max_length=10)
    month = models.CharField(max_length=10)
    time = models.IntegerField()
    primary_type_grouped = models.CharField(max_length=50)
    zone = models.CharField(max_length=10)
    season = models.CharField(max_length=10)
    loc_grouped = models.CharField(max_length=50)

    def __str__(self):
        return self.case_number
    
    class Meta:
        db_table = 'processed_crime_records'
        verbose_name_plural = 'Processed Crime Records'
        ordering = ('date',)

    def total_crimes(self):
        return self.objects.count()
    


class latest_model_statistics(models.Model):
    model_name = models.CharField(max_length=50, blank=True)
    model_accuracy = models.FloatField()
    model_precision = models.FloatField()
    model_recall = models.FloatField()
    model_error = models.CharField(max_length=250)
    model_f1_score = models.FloatField()
    model_confusion_matrix = models.CharField(max_length=1000)
    model_classification_report = models.CharField(max_length=1000)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.timestamp)
    
    def save(self, *args, **kwargs):
        if not self.model_name:
            self.model_name = 'Random Forest Classifier' + str(self.timestamp)
        super(latest_model_statistics, self).save(*args, **kwargs)
    
    class Meta:
        db_table = 'latest_model_statistics'
        verbose_name_plural = 'Latest Model Statistics'
        ordering = ('-timestamp',)

    def total_crimes(self):
        return self.objects.count()
    

class crime_type_model_statistics(models.Model):
    model_name = models.CharField(max_length=50, blank=True)
    model_accuracy = models.FloatField()
    model_precision = models.FloatField()
    model_recall = models.FloatField()
    model_error = models.CharField(max_length=250)
    model_f1_score = models.FloatField()
    model_confusion_matrix = models.CharField(max_length=1000)
    model_classification_report = models.CharField(max_length=1000)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.timestamp)
    
    def save(self, *args, **kwargs):
        if not self.model_name:
            self.model_name = 'Random Forest Classifier' + str(self.timestamp)
        super(crime_type_model_statistics, self).save(*args, **kwargs)
    
    class Meta:
        db_table = 'crime_type_model_statistics'
        verbose_name_plural = 'Crime Type Model Statistics'
        ordering = ('-timestamp',)

    def total_crimes(self):
        return self.objects.count() 
    

class latest_predictions_plots(models.Model):
    tree_plot = models.CharField(max_length=1000)
    kmeans_plot = models.CharField(max_length=1000)
    dbscan_plot = models.CharField(max_length=1000)
    hierarchical_plot = models.CharField(max_length=1000)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.timestamp)
    
    class Meta:
        db_table = 'latest_predictions_plots'
        verbose_name_plural = 'Latest Predictions Plots'
        ordering = ('-timestamp',)

    def total_crimes(self):
        return self.objects.count()

