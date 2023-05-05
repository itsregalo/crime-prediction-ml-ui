# Generated by Django 4.2 on 2023-05-05 23:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0007_remove_processedcrimedata_location"),
    ]

    operations = [
        migrations.CreateModel(
            name="crime_type_model_statistics",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("model_name", models.CharField(blank=True, max_length=50)),
                ("model_accuracy", models.FloatField()),
                ("model_precision", models.FloatField()),
                ("model_recall", models.FloatField()),
                ("model_error", models.FloatField()),
                ("model_f1_score", models.FloatField()),
                ("model_confusion_matrix", models.CharField(max_length=1000)),
                ("model_classification_report", models.CharField(max_length=1000)),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "verbose_name_plural": "Crime Type Model Statistics",
                "db_table": "crime_type_model_statistics",
                "ordering": ("-timestamp",),
            },
        ),
        migrations.CreateModel(
            name="latest_model_statistics",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("model_name", models.CharField(blank=True, max_length=50)),
                ("model_accuracy", models.FloatField()),
                ("model_precision", models.FloatField()),
                ("model_recall", models.FloatField()),
                ("model_error", models.FloatField()),
                ("model_f1_score", models.FloatField()),
                ("model_confusion_matrix", models.CharField(max_length=1000)),
                ("model_classification_report", models.CharField(max_length=1000)),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "verbose_name_plural": "Latest Model Statistics",
                "db_table": "latest_model_statistics",
                "ordering": ("-timestamp",),
            },
        ),
        migrations.CreateModel(
            name="latest_predictions_plots",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("tree_plot", models.CharField(max_length=1000)),
                ("kmeans_plot", models.CharField(max_length=1000)),
                ("dbscan_plot", models.CharField(max_length=1000)),
                ("hierarchical_plot", models.CharField(max_length=1000)),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "verbose_name_plural": "Latest Predictions Plots",
                "db_table": "latest_predictions_plots",
                "ordering": ("-timestamp",),
            },
        ),
    ]
