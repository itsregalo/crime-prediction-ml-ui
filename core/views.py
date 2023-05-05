from django.shortcuts import render
from django.contrib import messages
from tablib import Dataset
from .resources import CrimeResource
from django.contrib.auth.decorators import login_required
from IPython.display import HTML

# pagination
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


from .models import Crime

@login_required
def IndexView(request):
    data_count = Crime.objects.count()
    latest_crimes = Crime.objects.order_by('-date')[:15]
    context = {
        'data_count': data_count,
        'latest_crimes': latest_crimes,
    }
    return render(request, 'index.html', context)

def DataTableView(request):
    crimes = Crime.objects.all()
    paginator = Paginator(crimes, 100)
    page = request.GET.get('page')
    try:
        crimes = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        crimes = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        crimes = paginator.page(paginator.num_pages)
    context = {
        'crimes': crimes,
    }
    return render(request, 'data.html', context)


def import_data(request):
    if request.method == 'POST':
        file_format = request.POST['file-format']
        crime_resource = CrimeResource()
        dataset = Dataset()
        new_crimes = request.FILES['import-file']

        # Check the file format
        if file_format == 'csv':
            imported_data = dataset.load(new_crimes.read().decode('utf-8'), format=file_format)
        elif file_format == 'json':
            imported_data = dataset.load(new_crimes.read().decode('utf-8'), format=file_format)
        else:
            messages.error(request, "Unsupported file type.")
            return render(request, 'import_data.html')

        # Import the data
        result = crime_resource.import_data(dataset, dry_run=True)
        if not result.has_errors():
            crime_resource.import_data(dataset, dry_run=False)
            messages.success(request, "File imported successfully.")
        else:
            messages.error(request, "There was an error importing the file.")

    return render(request, 'import_data.html')

#Importing all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN
sns.set_style("darkgrid")

import io
import urllib, base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def data_description(request):
    """
    get data info, data description, data visualization
    """
    original_dataset = Crime.objects.all()
    df = pd.DataFrame(list(original_dataset.values()))
    df = df.drop(['id'], axis=1)

    # handling any inconsistent column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    # data info, present in the data_description.html
    data_info = df.info()
    data_description = df.describe()
    sample_data = df.head()
    # no of districts
    no_of_districts = df['district'].nunique()

    # Removing Primary key type attriburtes as they of no use for any type of analysis, Location columns is just a  combination of Latitude and Longitude

    df.drop(['case_number','location'],axis=1,inplace=True)

    # generate the plots
    fig = plt.figure(figsize=(15, 5))
    msno.heatmap(df, ax=fig.add_subplot(121))
    msno.dendrogram(df, ax=fig.add_subplot(122))
    
    # convert the plots to base64 encoded strings
    canvas = FigureCanvas(fig)
    msno_heatmap_png_output = io.BytesIO()
    canvas.print_png(msno_heatmap_png_output)
    msno_heatmap_png_output.seek(0)
    msno_heatmap_png_base64 = base64.b64encode(msno_heatmap_png_output.getvalue()).decode('utf-8').replace('\n', '')
    heatmap_image = "data:image/png;base64,{}".format(msno_heatmap_png_base64)

    canvas = FigureCanvas(fig)
    msno_dendrogram_png_output = io.BytesIO()
    canvas.print_png(msno_dendrogram_png_output)
    msno_dendrogram_png_output.seek(0)
    msno_dendrogram_png_base64 = base64.b64encode(msno_dendrogram_png_output.getvalue()).decode('utf-8').replace('\n', '')
    dendrogram_image = "data:image/png;base64,{}".format(msno_dendrogram_png_base64)


    """
    html context to display the plots in html

    <img src="{{ msno_heatmap }}" alt="msno_heatmap">
    """

    
    
    # to display the plots in h
    context = {
        'data_info': data_info,
        'data_description': data_description,
        'total_rows': df.shape[0],
        'missing_values': df.isnull().sum().sum(),
        'no_of_districts': no_of_districts,
        'sample_data': sample_data.to_html(classes='table table-striped table-hover'),
        # last_updated converted to datetime
        'last_updated': pd.to_datetime(df['updated_on']).max(),

        # plots to display in html
        'msno_heatmap': heatmap_image,
        'msno_dendrogram': dendrogram_image,
    }
    return render(request, 'data_description.html', context)
