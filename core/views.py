from django.shortcuts import render, HttpResponseRedirect, redirect
from django.contrib import messages
from tablib import Dataset
from .resources import CrimeResource
from django.contrib.auth.decorators import login_required
from IPython.display import HTML
from django.urls import reverse

# pagination
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


from .models import Crime, ProcessedCrimeData, latest_model_statistics, crime_type_model_statistics, latest_predictions_plots
# import settings
from django.conf import settings
# import pickle
import pickle
import os


@login_required
def IndexView(request):
    data_count = Crime.objects.count()
    cleaned_crimes = ProcessedCrimeData.objects.all()
    latest_crimes = Crime.objects.order_by('-date')[:15]
    latest_training_plot = latest_predictions_plots.objects.latest('timestamp')
    context = {
        'data_count': data_count,
        'latest_crimes': latest_crimes,
        'latest_training_plot': latest_training_plot,
        'cleaned_crimes_count': cleaned_crimes.count(),
    }
    return render(request, 'index.html', context)


def processed_data(request):
    processed_data = ProcessedCrimeData.objects.all()
    paginator = Paginator(processed_data, 100)
    page = request.GET.get('page')
    try:
        processed_data = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        processed_data = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        processed_data = paginator.page(paginator.num_pages)
    context = {
        'processed_data': processed_data,
    }
    return render(request, 'processed_data.html', context)

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



def plot_to_base64(plt):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode('utf-8')
    return b64

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
        'sample_data': sample_data.to_html(classes='table table-striped table-bordered'),
        # last_updated converted to datetime
        'last_updated': pd.to_datetime(df['updated_on']).max(),

        # plots to display in html
        'msno_heatmap': heatmap_image,
        'msno_dendrogram': dendrogram_image,
    }
    return render(request, 'data_description.html', context)


def clean_data(request, *args, **kwargs):
    """
    get data info, data description, data visualization
    """
    original_dataset = Crime.objects.all()
    crimes_data = pd.DataFrame(list(original_dataset.values()))
    crimes_data = crimes_data.drop(['id'], axis=1)

    #Handling any inconsistensis of column names
    crimes_data.columns = crimes_data.columns.str.strip()
    crimes_data.columns = crimes_data.columns.str.replace(',', '')
    crimes_data.columns = crimes_data.columns.str.replace(' ', '_')
    crimes_data.columns = crimes_data.columns.str.lower()

    # Removing Primary key type attriburtes as they of no use for any type of analysis, Location columns is just a 
    # combination of Latitude and Longitude
    crimes_data.drop(['case_number','location'],axis=1,inplace=True)

    #Dropping observations where latitude is null/Nan
    crimes_data.dropna(subset=['latitude'],inplace=True)
    crimes_data.reset_index(drop=True,inplace=True)

    crimes_data.dropna(inplace=True)
    crimes_data.reset_index(drop=True,inplace=True)

    #Converting the data column to datetime object so we can get better results of our analysis
    #Get the day of the week,month and time of the crimes
    crimes_data.date = pd.to_datetime(crimes_data.date)
    crimes_data['day_of_week'] = crimes_data.date.dt.day_name()
    crimes_data['month'] = crimes_data.date.dt.month_name()
    crimes_data['time'] = crimes_data.date.dt.hour

    #Mapping similar crimes under one group.
    primary_type_map = {
        ('BURGLARY','MOTOR VEHICLE THEFT','THEFT','ROBBERY') : 'THEFT',
        ('BATTERY','ASSAULT','NON-CRIMINAL','NON-CRIMINAL (SUBJECT SPECIFIED)') : 'NON-CRIMINAL_ASSAULT',
        ('CRIM SEXUAL ASSAULT','SEX OFFENSE','STALKING','PROSTITUTION') : 'SEXUAL_OFFENSE',
        ('WEAPONS VIOLATION','CONCEALED CARRY LICENSE VIOLATION') :  'WEAPONS_OFFENSE',
        ('HOMICIDE','CRIMINAL DAMAGE','DECEPTIVE PRACTICE','CRIMINAL TRESPASS') : 'CRIMINAL_OFFENSE',
        ('KIDNAPPING','HUMAN TRAFFICKING','OFFENSE INVOLVING CHILDREN') : 'HUMAN_TRAFFICKING_OFFENSE',
        ('NARCOTICS','OTHER NARCOTIC VIOLATION') : 'NARCOTIC_OFFENSE',
        ('OTHER OFFENSE','ARSON','GAMBLING','PUBLIC PEACE VIOLATION','INTIMIDATION','INTERFERENCE WITH PUBLIC OFFICER','LIQUOR LAW VIOLATION','OBSCENITY','PUBLIC INDECENCY') : 'OTHER_OFFENSE'
    }
    primary_type_mapping = {}
    for keys, values in primary_type_map.items():
        for key in keys:
            primary_type_mapping[key] = values
    crimes_data['primary_type_grouped'] = crimes_data.primary_type.map(primary_type_mapping)

    #Zone where the crime has occured
    zone_mapping = {
        'N' : 'North',
        'S' : 'South',
        'E' : 'East',
        'W' : 'West'
    }
    crimes_data['zone'] = crimes_data.block.str.split(" ", n = 2, expand = True)[1].map(zone_mapping)

    #Mapping seasons from month of crime
    season_map = {
        ('March','April','May') : 'Spring',
        ('June','July','August') : 'Summer',
        ('September','October','November') : 'Fall',
        ('December','January','February') : 'Winter'
    }
    season_mapping = {}
    for keys, values in season_map.items():
        for key in keys:
            season_mapping[key] = values
    crimes_data['season'] = crimes_data.month.map(season_mapping)

    #Mapping similar locations of crime under one group.
    loc_map = {
        ('RESIDENCE', 'APARTMENT', 'CHA APARTMENT', 'RESIDENCE PORCH/HALLWAY', 'RESIDENCE-GARAGE',
        'RESIDENTIAL YARD (FRONT/BACK)', 'DRIVEWAY - RESIDENTIAL', 'HOUSE') : 'RESIDENCE',
        
        ('BARBERSHOP', 'COMMERCIAL / BUSINESS OFFICE', 'CURRENCY EXCHANGE', 'DEPARTMENT STORE', 'RESTAURANT',
        'ATHLETIC CLUB', 'TAVERN/LIQUOR STORE', 'SMALL RETAIL STORE', 'HOTEL/MOTEL', 'GAS STATION',
        'AUTO / BOAT / RV DEALERSHIP', 'CONVENIENCE STORE', 'BANK', 'BAR OR TAVERN', 'DRUG STORE',
        'GROCERY FOOD STORE', 'CAR WASH', 'SPORTS ARENA/STADIUM', 'DAY CARE CENTER', 'MOVIE HOUSE/THEATER',
        'APPLIANCE STORE', 'CLEANING STORE', 'PAWN SHOP', 'FACTORY/MANUFACTURING BUILDING', 'ANIMAL HOSPITAL',
        'BOWLING ALLEY', 'SAVINGS AND LOAN', 'CREDIT UNION', 'KENNEL', 'GARAGE/AUTO REPAIR', 'LIQUOR STORE',
        'GAS STATION DRIVE/PROP.', 'OFFICE', 'BARBER SHOP/BEAUTY SALON') : 'BUSINESS',
        
        ('VEHICLE NON-COMMERCIAL', 'AUTO', 'VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)', 'TAXICAB',
        'VEHICLE-COMMERCIAL', 'VEHICLE - DELIVERY TRUCK', 'VEHICLE-COMMERCIAL - TROLLEY BUS',
        'VEHICLE-COMMERCIAL - ENTERTAINMENT/PARTY BUS') : 'VEHICLE',
        
        ('AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA', 'CTA PLATFORM', 'CTA STATION', 'CTA BUS STOP',
        'AIRPORT TERMINAL UPPER LEVEL - SECURE AREA', 'CTA TRAIN', 'CTA BUS', 'CTA GARAGE / OTHER PROPERTY',
        'OTHER RAILROAD PROP / TRAIN DEPOT', 'AIRPORT TERMINAL LOWER LEVEL - SECURE AREA',
        'AIRPORT BUILDING NON-TERMINAL - SECURE AREA', 'AIRPORT EXTERIOR - NON-SECURE AREA', 'AIRCRAFT',
        'AIRPORT PARKING LOT', 'AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA', 'OTHER COMMERCIAL TRANSPORTATION',
        'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA', 'AIRPORT VENDING ESTABLISHMENT',
        'AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA', 'AIRPORT EXTERIOR - SECURE AREA', 'AIRPORT TRANSPORTATION SYSTEM (ATS)',
        'CTA TRACKS - RIGHT OF WAY', 'AIRPORT/AIRCRAFT', 'BOAT/WATERCRAFT', 'CTA PROPERTY', 'CTA "L" PLATFORM',
        'RAILROAD PROPERTY') : 'PUBLIC_TRANSPORTATION',
        
        ('HOSPITAL BUILDING/GROUNDS', 'NURSING HOME/RETIREMENT HOME', 'SCHOOL, PUBLIC, BUILDING',
        'CHURCH/SYNAGOGUE/PLACE OF WORSHIP', 'SCHOOL, PUBLIC, GROUNDS', 'SCHOOL, PRIVATE, BUILDING',
        'MEDICAL/DENTAL OFFICE', 'LIBRARY', 'COLLEGE/UNIVERSITY RESIDENCE HALL', 'YMCA', 'HOSPITAL') : 'PUBLIC_BUILDING',
        
        ('STREET', 'PARKING LOT/GARAGE(NON.RESID.)', 'SIDEWALK', 'PARK PROPERTY', 'ALLEY', 'CEMETARY',
        'CHA HALLWAY/STAIRWELL/ELEVATOR', 'CHA PARKING LOT/GROUNDS', 'COLLEGE/UNIVERSITY GROUNDS', 'BRIDGE',
        'SCHOOL, PRIVATE, GROUNDS', 'FOREST PRESERVE', 'LAKEFRONT/WATERFRONT/RIVERBANK', 'PARKING LOT', 'DRIVEWAY',
        'HALLWAY', 'YARD', 'CHA GROUNDS', 'RIVER BANK', 'STAIRWELL', 'CHA PARKING LOT') : 'PUBLIC_AREA',
        
        ('POLICE FACILITY/VEH PARKING LOT', 'GOVERNMENT BUILDING/PROPERTY', 'FEDERAL BUILDING', 'JAIL / LOCK-UP FACILITY',
        'FIRE STATION', 'GOVERNMENT BUILDING') : 'GOVERNMENT',
        
        ('OTHER', 'ABANDONED BUILDING', 'WAREHOUSE', 'ATM (AUTOMATIC TELLER MACHINE)', 'VACANT LOT/LAND',
        'CONSTRUCTION SITE', 'POOL ROOM', 'NEWSSTAND', 'HIGHWAY/EXPRESSWAY', 'COIN OPERATED MACHINE', 'HORSE STABLE',
        'FARM', 'GARAGE', 'WOODED AREA', 'GANGWAY', 'TRAILER', 'BASEMENT', 'CHA PLAY LOT') : 'OTHER'  
    }

    loc_mapping = {}
    for keys, values in loc_map.items():
        for key in keys:
            loc_mapping[key] = values
    crimes_data['loc_grouped'] = crimes_data.location_description.map(loc_mapping)

    #Mapping crimes to ints to get better information from plots
    crimes_data.arrest = crimes_data.arrest.astype(int)
    crimes_data.domestic = crimes_data.domestic.astype(int)

    # save the cleaned data to the ProcessedCrimeData model
    from .models import ProcessedCrimeData

    crimes = ProcessedCrimeData.objects.all()
    crimes.delete()

    for index, row in crimes_data.iterrows():
        crime = ProcessedCrimeData(
            date = row['date'],
            block = row['block'],
            iucr = row['iucr'],
            primary_type = row['primary_type'],
            description = row['description'],
            location_description = row['location_description'],
            arrest = row['arrest'],
            domestic = row['domestic'],
            beat = row['beat'],
            district = row['district'],
            ward = row['ward'],
            community_area = row['community_area'],
            fbi_code = row['fbi_code'],
            x_coordinate = row['x_coordinate'],
            y_coordinate = row['y_coordinate'],
            year = row['year'],
            updated_on = row['updated_on'],
            latitude = row['latitude'],
            longitude = row['longitude'],
            day_of_week = row['day_of_week'],
            month = row['month'],
            time = row['time'],
            primary_type_grouped = row['primary_type_grouped'],
            zone = row['zone'],
            season = row['season'],
            loc_grouped = row['loc_grouped']
        )
        crime.save()

    context = {
        'crimes_data': crimes_data
    }

    return render(request, 'preprocessing.html', context)

def data_analytics(request, *args, **kwargs):
    crimes_data = ProcessedCrimeData.objects.all()
    sample_dataset = crimes_data[:10]


    # convert to dataframe
    crimes_data_df = pd.DataFrame(list(crimes_data.values()))

    # 1. Crimes per day of the week
    fig, ax = plt.subplots(figsize=(9, 6))
    crimes_data_df.groupby('day_of_week').size().plot(kind='bar', ax=ax)
    ax.set_title('Crimes per day of the week')
    ax.set_xlabel('Day of the week')
    ax.set_ylabel('Number of crimes')
    crimes_per_day_b64 = plot_to_base64(plt)

    # 2. Crimes per month
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(data=crimes_data_df,x='month',hue='year',order=crimes_data_df.month.value_counts().index,palette='Set2')
    ax.set_title('Crimes per month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of crimes')
    crimes_per_month_b64 = plot_to_base64(plt)

    # 3. Arrests per month
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.pointplot(data=crimes_data_df,x='month',y='arrest',hue='year',order=crimes_data_df.month.value_counts().index,palette='Set2')
    ax.set_title('Arrests per month')
    ax.set_xlabel('Arrest')
    ax.set_ylabel('Month')
    arrests_per_month_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(data=crimes_data_df,x='zone',hue='year',order=crimes_data_df.zone.value_counts().index,palette='Set2')
    ax.set_title('Crimes per zone')
    ax.set_xlabel('Zone')
    ax.set_ylabel('Number of crimes')
    crimes_per_zone_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(data=crimes_data_df,x='season',hue='year',palette='Set2')
    ax.set_title('Crimes per season')
    ax.set_xlabel('Season')
    ax.set_ylabel('Number of crimes')
    crimes_per_season_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(data=crimes_data_df,x='year',hue='arrest',palette='Set2')
    ax.set_title('Arrests per year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of arrests')
    arrests_per_year_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    plt.pie(crimes_data_df.primary_type_grouped.value_counts(),labels=crimes_data_df.primary_type_grouped.value_counts().index,autopct='%1.1f%%',shadow=True,radius=2.5)
    ax.set_title('Crimes per type')
    ax.set_xlabel('Type')
    ax.set_ylabel('Number of crimes')
    plt.legend(loc = 'best')
    crimes_per_type_pie = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    plt.pie(crimes_data_df.loc_grouped.value_counts(),labels=crimes_data_df.loc_grouped.value_counts().index,autopct='%1.1f%%',shadow=True,radius=2.5)
    plt.legend(loc = 'best')
    ax.set_title('Crimes per location')
    ax.set_xlabel('Location')
    ax.set_ylabel('Number of crimes')
    crimes_per_loc_pie = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(data=crimes_data_df,x='day_of_week',order=crimes_data_df.day_of_week.value_counts().index,palette='Set2')
    ax.set_title('Crimes per day of the week')
    ax.set_xlabel('Day of the week')
    ax.set_ylabel('Number of crimes')
    complete_crimes_per_day_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(data=crimes_data_df,x='month',order=crimes_data_df.month.value_counts().index,palette='Set2')
    ax.set_title('Crimes per month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of crimes')
    complete_crimes_per_month_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.pointplot(data=crimes_data_df,x=crimes_data_df.time.value_counts().index,y=crimes_data_df.time.value_counts())
    ax.set_title('Crimes per time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of crimes')
    complete_crimes_per_time_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(data=crimes_data_df,x='zone',order=crimes_data_df.zone.value_counts().index,palette='Set2')
    ax.set_title('Crimes per zone')
    ax.set_xlabel('Zone')
    ax.set_ylabel('Number of crimes')
    complete_crimes_per_zone_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(data=crimes_data_df,x='season',order=crimes_data_df.season.value_counts().index,palette='Set2')
    ax.set_title('Crimes per season')
    ax.set_xlabel('Season')
    ax.set_ylabel('Number of crimes')
    complete_crimes_per_season_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(data=crimes_data_df,x='arrest',hue='primary_type_grouped',palette='Set2')
    ax.set_title('Arrests per crime type')
    ax.set_xlabel('Arrest')
    ax.set_ylabel('Number of arrests')
    plt.legend(loc = 'best')
    complete_arrests_per_crime_type_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=crimes_data_df,x=crimes_data_df.location_description.value_counts()[0:20].index,y=crimes_data_df.location_description.value_counts()[0:20].values,palette='Set2')
    ax.set_title('Crimes per location')
    ax.set_xlabel('Location')
    ax.set_ylabel('Number of crimes')
    plt.xticks(rotation=45)
    complete_crimes_per_location_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(9, 6))
    new_crimes_data_df = crimes_data_df.loc[(crimes_data_df['x_coordinate'] > 0) & (crimes_data_df['y_coordinate'] > 0)]
    sns.lmplot(x='x_coordinate',
            y='y_coordinate',
            data=new_crimes_data_df,
            fit_reg=False,
            hue='primary_type_grouped',
            scatter_kws={"marker": "D", 
                            "s": 20})
    ax = plt.gca()
    ax.set_title('Chicago Crime Map')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.legend(loc = 'best')
    complete_crimes_map_b64 = plot_to_base64(plt)

    fig, ax = plt.subplots(figsize=(10, 6))
    crimes_data_df.loc[(crimes_data_df['x_coordinate'] > 0) & (crimes_data_df['y_coordinate'] > 0)]
    sns.lmplot(x='x_coordinate',
            y='y_coordinate',
            data=new_crimes_data_df,
            fit_reg=False,
            hue='primary_type_grouped',
            scatter_kws={"marker": "D",
                        "s": 20})
    ax = plt.gca()
    ax.set_title('Chicago Crime Map')
    complete_crimes_map_b64_2 = plot_to_base64(plt)

    context = {
        'sample_dataset': sample_dataset,
        'total_crimes': crimes_data.count(),
        'crimes_data_df': crimes_data_df,
        'crimes_per_day_b64': crimes_per_day_b64,
        'crimes_per_month_b64': crimes_per_month_b64,
        'arrests_per_month_b64': arrests_per_month_b64,
        'crimes_per_zone_b64': crimes_per_zone_b64,
        'crimes_per_season_b64': crimes_per_season_b64,
        'arrests_per_year_b64': arrests_per_year_b64,
        'crimes_per_type_pie': crimes_per_type_pie,
        'crimes_per_loc_pie': crimes_per_loc_pie,
        'complete_crimes_per_day_b64': complete_crimes_per_day_b64,
        'complete_crimes_per_month_b64': complete_crimes_per_month_b64,
        'complete_crimes_per_time_b64': complete_crimes_per_time_b64,
        'complete_crimes_per_zone_b64': complete_crimes_per_zone_b64,
        'complete_crimes_per_season_b64': complete_crimes_per_season_b64,
        'complete_arrests_per_crime_type_b64': complete_arrests_per_crime_type_b64,
        'complete_crimes_per_location_b64': complete_crimes_per_location_b64,
        'complete_crimes_map_b64': complete_crimes_map_b64,
        'complete_crimes_map_b64_2': complete_crimes_map_b64_2,
    }

    return render(request, 'data_analytics.html', context)


def verify_cleaning(request, *args, **kwargs):
    if request.method == 'POST':
        return HttpResponseRedirect(redirect('core:preprocess-data'))


def train_model(request, *args, **kwargs):
    cleaned_data = ProcessedCrimeData.objects.all()

    # convert to pandas dataframe
    crimes_data = pd.DataFrame(list(cleaned_data.values()))

    # drop unnecessary columns
    crimes_data.drop(['id'], axis=1, inplace=True)

    #Converting the numercial attributes to categorical attributes
    crimes_data.year = pd.Categorical(crimes_data.year)
    crimes_data.time = pd.Categorical(crimes_data.time)
    crimes_data.domestic = pd.Categorical(crimes_data.domestic)
    crimes_data.arrest = pd.Categorical(crimes_data.arrest)
    crimes_data.beat = pd.Categorical(crimes_data.beat)
    crimes_data.district = pd.Categorical(crimes_data.district)
    crimes_data.ward = pd.Categorical(crimes_data.ward)
    crimes_data.community_area = pd.Categorical(crimes_data.community_area)

    crimes_data_prediction = crimes_data.drop(['date','block','iucr','primary_type','description','location_description','fbi_code','updated_on','x_coordinate','y_coordinate'],axis=1)

    crimes_data_prediction = pd.get_dummies(crimes_data_prediction,drop_first=True)

    #Train test split with a test set size of 30% of entire data
    X_train, X_test, y_train, y_test = train_test_split(crimes_data_prediction.drop(['arrest_1'],axis=1),crimes_data_prediction['arrest_1'], test_size=0.3, random_state=42)

    #Standardizing the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train) 
    X_test = scaler.transform(X_test)

    #Training the model
    #Random Forest classifier  - Best one
    model = RandomForestClassifier(n_estimators = 10,criterion='entropy',random_state=42)

    #Fitting the model
    model.fit(X_train,y_train)

    #Predicting the test set results
    y_pred = model.predict(X_test)

    # save the model to static folder
    filename = 'finalized_model.sav'
    pickle.dump(model, open(os.path.join(settings.STATIC_ROOT, filename), 'wb'))

    # Compute confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot confusion matrix
    sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')
    plt.tight_layout()
    model_confusion_matrix_b64 = plot_to_base64(plt)

    #Classification Metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    error = 1 - accuracy,
    precision = metrics.precision_score(y_test, y_pred,)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    classification_report = metrics.classification_report(y_test, y_pred)


    latest_stats = latest_model_statistics.objects.create(
    model_accuracy=accuracy,
    model_precision=precision,
    model_error=error,
    model_recall=recall,
    model_f1_score=f1_score,
    model_classification_report=classification_report,
    model_confusion_matrix=conf_matrix,
    training_data_size=len(X_train),
    test_data_size=len(X_test),
    model_confusion_matrix_plot=model_confusion_matrix_b64,
    )
    latest_stats.save()



    crimes_data_type = crimes_data.loc[crimes_data.primary_type_grouped.isin(['THEFT','NON-CRIMINAL_ASSAULT','CRIMINAL_OFFENSE'])]
    crimes_data_prediction = crimes_data_type.drop(['date','block','iucr','primary_type','description','location_description','fbi_code','updated_on','x_coordinate','y_coordinate','primary_type_grouped'],axis=1)
    crimes_data_prediction_type = crimes_data_type.primary_type_grouped
    crimes_data_prediction = pd.get_dummies(crimes_data_prediction,drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(crimes_data_prediction,crimes_data_prediction_type, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train) 
    X_test = scaler.transform(X_test)

    #Random Forest classifier for type of crime
    model = RandomForestClassifier(n_estimators = 10,criterion='entropy',random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    # Compute confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot confusion matrix
    sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')
    plt.tight_layout()
    crime_type_model_confusion_matrix_b64_2 = plot_to_base64(plt)

    #Classification Metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    error = 1 - accuracy,
    precision = metrics.precision_score(y_test, y_pred,average='weighted')
    recall = metrics.recall_score(y_test, y_pred,average='weighted')
    f1_score = metrics.f1_score(y_test, y_pred,average='weighted')
    classification_report = metrics.classification_report(y_test, y_pred)

    try:
        crime_type_latest_stats = crime_type_model_statistics.objects.create(
        model_accuracy=accuracy,
        model_precision=precision,
        model_error=error,
        model_recall=recall,
        model_f1_score=f1_score,
        model_classification_report=classification_report,
        model_confusion_matrix=conf_matrix,
        training_data_size=len(X_train),
        test_data_size=len(X_test),
        model_confusion_matrix_plot=crime_type_model_confusion_matrix_b64_2,
        )
        crime_type_latest_stats.save()
    except:
        pass

    # Calculated the number of occrurances for each type of crime category in each district
    district_crime_rates = pd.DataFrame(columns=['theft_count', 'assault_count', 'sexual_offense_count', 
                                                'weapons_offense_count', 'criminal_offense_count', 
                                                'human_trafficking_count', 'narcotic_offense_count', 
                                                'other_offense_count'])
    district_crime_rates = district_crime_rates.astype(int) 

    for i in range(1, 32):   
        temp_district_df = crimes_data[crimes_data['district'] == i] 

        temp_district_theft = temp_district_df[temp_district_df['primary_type_grouped'] == 'THEFT'] 
        num_theft = temp_district_theft.primary_type_grouped.count() 
        
        temp_district_assault = temp_district_df[temp_district_df['primary_type_grouped'] == 'NON-CRIMINAL_ASSAULT'] 
        num_assault = temp_district_assault.primary_type_grouped.count()    
        
        temp_district_sexual_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'SEXUAL_OFFENSE'] 
        num_sexual_offense = temp_district_sexual_offense.primary_type_grouped.count()
        
        temp_district_weapons_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'WEAPONS_OFFENSE'] 
        num_weapons_offense = temp_district_weapons_offense.primary_type_grouped.count()
        
        temp_district_criminal_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'CRIMINAL_OFFENSE'] 
        num_criminal_offense = temp_district_criminal_offense.primary_type_grouped.count()
        
        temp_district_human_trafficking = temp_district_df[temp_district_df['primary_type_grouped'] == 'HUMAN_TRAFFICKING_OFFENSE'] 
        num_human_trafficking = temp_district_human_trafficking.primary_type_grouped.count()
        
        temp_district_narcotic_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'NARCOTIC_OFFENSE'] 
        num_narcotic_offense = temp_district_narcotic_offense.primary_type_grouped.count()
        
        temp_district_other_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'OTHER_OFFENSE'] 
        num_other_offense = temp_district_other_offense.primary_type_grouped.count()

        district_crime_rates.loc[i] = [num_theft, num_assault, num_sexual_offense, num_weapons_offense, num_criminal_offense, num_human_trafficking, num_narcotic_offense, num_other_offense]    
        
    # Standardize the data
    district_crime_rates_standardized = preprocessing.scale(district_crime_rates)
    district_crime_rates_standardized = pd.DataFrame(district_crime_rates_standardized)

    # Clustering with K-Means 
    kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(district_crime_rates_standardized)
    #y_kmeans

    #beginning of  the cluster numbering with 1 instead of 0
    y_kmeans1=y_kmeans+1

    # New list called cluster
    kmeans_clusters = list(y_kmeans1)
    # Adding cluster to our data set
    district_crime_rates['kmeans_cluster'] = kmeans_clusters

    #Mean of clusters 1 to 4
    kmeans_mean_cluster = pd.DataFrame(round(district_crime_rates.groupby('kmeans_cluster').mean(),1))

    # Clustering with DBSCAN
    clustering = DBSCAN(eps = 1, min_samples = 3, metric = "euclidean").fit(district_crime_rates_standardized)

    # Show clusters
    dbscan_clusters = clustering.labels_
    # print(clusters)

    district_crime_rates['dbscan_clusters'] = dbscan_clusters + 2
    #district_crime_rates.head()

    # Clustering with Hierarchical Clustering with average linkage
    clustering = linkage(district_crime_rates_standardized, method = "average", metric = "euclidean")

    # Plot dendrogram
    plt.figure()
    dendrogram(clustering)  
    hierarchial_dendrogram_b64 = plot_to_base64(plt) # plot to base64


    # Form clusters
    hierarchical_clusters = fcluster(clustering, 4, criterion = 'maxclust')
    # print(clusters)

    district_crime_rates['hierarchical_clusters'] = hierarchical_clusters 

    # Add 'district' column
    district_crime_rates['district'] = district_crime_rates.index
    district_crime_rates = district_crime_rates[['district', 'kmeans_cluster', 'dbscan_clusters', 'hierarchical_clusters', 'theft_count', 'assault_count', 'sexual_offense_count', 'weapons_offense_count', 'criminal_offense_count', 'human_trafficking_count', 'narcotic_offense_count', 'other_offense_count']]

    # Remove all columns but 'district' & each method's cluster
    district_crime_rates = district_crime_rates.drop(['theft_count', 'assault_count', 'sexual_offense_count', 'weapons_offense_count', 'criminal_offense_count', 'human_trafficking_count', 'narcotic_offense_count', 'other_offense_count'], axis=1)

    # Merge each district's clusters for each method into a single dataframe 
    crimes_data_clustered = pd.merge(crimes_data, district_crime_rates, on='district', how='inner')

    # Plot Crime level clusters by district (KMeans Clustering)
    new_crimes_data = crimes_data_clustered.loc[(crimes_data_clustered['x_coordinate']!=0)]

    plt.figure(figsize=(20,10))
    sns.lmplot(x='x_coordinate', y='y_coordinate', data=new_crimes_data, fit_reg=False, hue='kmeans_cluster', legend=True, scatter_kws={"s": 10})
    plt.title('Crime level clusters by district (KMeans Clustering)')
    crime_level_clusters_by_district_kmeans_b64 = plot_to_base64(plt) # plot to base64

    # Crime level clusters by district (DBScan Clustering)
    new_crimes_data = crimes_data_clustered.loc[(crimes_data_clustered['x_coordinate']!=0)]

    plt.figure(figsize=(20,10))
    sns.lmplot(x='x_coordinate', y='y_coordinate', data=new_crimes_data, fit_reg=False, hue='dbscan_clusters', legend=True, scatter_kws={"s": 10})
    plt.title('Crime level clusters by district (DBScan Clustering)')
    plt.legend(loc='upper right')
    dbscan_clusters_b64 = plot_to_base64(plt) # plot to base64

    # Crime level clusters by district (Hierarchical Clustering)
    new_crimes_data = crimes_data_clustered.loc[(crimes_data_clustered['x_coordinate']!=0)]
    sns.lmplot(x='x_coordinate', y='y_coordinate', data=new_crimes_data, fit_reg=False, hue='hierarchical_clusters', legend=True, scatter_kws={"s": 10})
    plt.title('Crime level clusters by district (Hierarchical Clustering)')
    hierarchical_clusters_b64 = plot_to_base64(plt) # plot to base64

    try:
        latest_plots = latest_predictions_plots.objects.create(
            tree_plot = hierarchial_dendrogram_b64,
            kmeans_plot = crime_level_clusters_by_district_kmeans_b64,
            dbscan_plot = dbscan_clusters_b64,
            hierarchical_plot = hierarchical_clusters_b64
        )
        latest_plots.save()
    except:
        pass

    return HttpResponseRedirect(reverse('core:nerd-stats'))
            

def nerd_statistics(request):
    latest_models_stats = latest_model_statistics.objects.all().order_by('-id')[0]
    crime_type_model_stats = crime_type_model_statistics.objects.all().order_by('-id')[0]

    context = {
        'latest_models_stats': latest_models_stats,
        'crime_type_model_stats': crime_type_model_stats
    }

    return render(request, 'nerd_statistics.html', context)