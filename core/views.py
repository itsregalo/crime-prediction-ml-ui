from django.shortcuts import render, HttpResponseRedirect, redirect
from django.contrib import messages
from tablib import Dataset
from .resources import CrimeResource
from django.contrib.auth.decorators import login_required
from IPython.display import HTML

# pagination
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


from .models import Crime, ProcessedCrimeData


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


    }

    return render(request, 'data_analytics.html', context)


def verify_cleaning(request, *args, **kwargs):
    if request.method == 'POST':
        return HttpResponseRedirect(redirect('core:preprocess-data'))