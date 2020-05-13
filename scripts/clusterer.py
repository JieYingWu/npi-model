from os.path import join, exists
import os
from string import capwords
import numpy as np
import datetime as dt
import pandas as pd
import umap
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from scipy.signal import savgol_filter
import seaborn as sns
from plotly import graph_objects as go

class DashboardData(object):
  data_dir = './data'
  converters = {'FIPS': lambda x: str(x).zfill(5)}

  columns_to_include = {
    "FIPS",
    "State",
    "Area_Name",
    "Rural-urban_Continuum Code_2013",
    "Urban_Influence_Code_2013",
    "Economic_typology_2015",
    "POP_ESTIMATE_2018",
    "N_POP_CHG_2018",
    "Births_2018",
    "Deaths_2018",
    "NATURAL_INC_2018",
    "INTERNATIONAL_MIG_2018",
    "DOMESTIC_MIG_2018",
    "NET_MIG_2018",
    # "RESIDUAL_2018",
    # "GQ_ESTIMATES_2018",
    # "R_birth_2018",
    # "R_death_2018",
    # "R_NATURAL_INC_2018",
    # "R_INTERNATIONAL_MIG_2018",
    # "R_DOMESTIC_MIG_2018",
    # "R_NET_MIG_2018",
    "Less than a high school diploma 2014-18",
    "High school diploma only 2014-18",
    "Some college or associate's degree 2014-18",
    "Bachelor's degree or higher 2014-18",
    "Percent of adults with less than a high school diploma 2014-18",
    "Percent of adults with a high school diploma only 2014-18",
    "Percent of adults completing some college or associate's degree 2014-18",
    "Percent of adults with a bachelor's degree or higher 2014-18",
    "POVALL_2018",
    "CI90LBAll_2018",
    "CI90UBALL_2018",
    "PCTPOVALL_2018",
    "CI90LBALLP_2018",
    "CI90UBALLP_2018",
    "POV017_2018",
    "CI90LB017_2018",
    "CI90UB017_2018",
    "PCTPOV017_2018",
    "CI90LB017P_2018",
    "CI90UB017P_2018",
    "POV517_2018",
    "CI90LB517_2018",
    "CI90UB517_2018",
    "PCTPOV517_2018",
    "CI90LB517P_2018",
    "CI90UB517P_2018",
    "MEDHHINC_2018",
    "CI90LBINC_2018",
    "CI90UBINC_2018",
    "Civilian_labor_force_2018",
    "Employed_2018",
    "Unemployed_2018",
    # "Unemployment_rate_2018",
    "Median_Household_Income_2018",
    # "Med_HH_Income_Percent_of_State_Total_2018",
    "Jan Precipitation / inch",
    "Feb Precipitation / inch",
    "Mar Precipitation / inch",
    "Apr Precipitation / inch",
    "May Precipitation / inch",
    "Jun Precipitation / inch",
    "Jul Precipitation / inch",
    "Aug Precipitation / inch",
    "Sep Precipitation / inch",
    "Oct Precipitation / inch",
    "Nov Precipitation / inch",
    "Dec Precipitation / inch",
    "Jan Temp AVG / F",
    "Feb Temp AVG / F",
    "Mar Temp AVG / F",
    "Apr Temp AVG / F",
    "May Temp AVG / F",
    "Jun Temp AVG / F",
    "Jul Temp AVG / F",
    "Aug Temp AVG / F",
    "Sep Temp AVG / F",
    "Oct Temp AVG / F",
    "Nov Temp AVG / F",
    "Dec Temp AVG / F",
    "Jan Temp Min / F",
    "Feb Temp Min / F",
    "Mar Temp Min / F",
    "Apr Temp Min / F",
    "May Temp Min / F",
    "Jun Temp Min / F",
    "Jul Temp Min / F",
    "Aug Temp Min / F",
    "Sep Temp Min / F",
    "Oct Temp Min / F",
    "Nov Temp Min / F",
    "Dec Temp Min / F",
    "Jan Temp Max / F",
    "Feb Temp Max / F",
    "Mar Temp Max / F",
    "Apr Temp Max / F",
    "May Temp Max / F",
    "Jun Temp Max / F",
    "Jul Temp Max / F",
    "Aug Temp Max / F",
    "Sep Temp Max / F",
    "Oct Temp Max / F",
    "Nov Temp Max / F",
    "Dec Temp Max / F",
    "Housing units",
    "Area in square miles - Total area",
    "Area in square miles - Water area",
    "Area in square miles - Land area",
    # "Density per square mile of land area - Population",
    # "Density per square mile of land area - Housing units",
    "Total_Male",
    "Total_Female",
    "Total_age0to17",
    "Male_age0to17",
    "Female_age0to17",
    "Total_age18to64",
    "Male_age18to64",
    "Female_age18to64",
    "Total_age65plus",
    "Male_age65plus",
    "Female_age65plus",
    "Total_age85plusr",
    "Male_age85plusr",
    "Female_age85plusr",
    "Total households",
    "Total households!!Family households (families)",
    "Total households!!Family households (families)!!With own children of the householder under 18 years",
    "Total households!!Family households (families)!!Married-couple family",
    "Total households!!Family households (families)!!Married-couple family!!With own children of the householder under 18 years",
    "Total households!!Family households (families)!!Male householder no wife present family",
    "HOUSEHOLDS BY TYPE!!",
    "Total households!!Family households (families)!!Female householder no husband present family",
    "Total households!!Family households (families)!!Female householder no husband present family!!With own children of the householder under 18 years",
    "Total households!!Nonfamily households",
    "Total households!!Nonfamily households!!Householder living alone",
    "Total households!!Nonfamily households!!Householder living alone!!65 years and over",
    "Total households!!Households with one or more people under 18 years",
    "Total households!!Households with one or more people 65 years and over",
    "Total households!!Average household size",
    "Total households!!Average family size",
    "RELATIONSHIP!!Population in households",
    "RELATIONSHIP!!Population in households!!Householder",
    "RELATIONSHIP!!Population in households!!Spouse",
    "RELATIONSHIP!!Population in households!!Child",
    "RELATIONSHIP!!Population in households!!Other relatives",
    "RELATIONSHIP!!Population in households!!Nonrelatives",
    "RELATIONSHIP!!Population in households!!Nonrelatives!!Unmarried partner",
    "MARITAL STATUS!!Males 15 years and over",
    "MARITAL STATUS!!Males 15 years and over!!Never married",
    "MARITAL STATUS!!Males 15 years and over!!Now married except separated",
    "MARITAL STATUS!!Males 15 years and over!!Separated",
    "MARITAL STATUS!!Males 15 years and over!!Widowed",
    "MARITAL STATUS!!Males 15 years and over!!Divorced",
    "MARITAL STATUS!!Females 15 years and over",
    "MARITAL STATUS!!Females 15 years and over!!Never married",
    "MARITAL STATUS!!Females 15 years and over!!Now married except separated",
    "MARITAL STATUS!!Females 15 years and over!!Separated",
    "MARITAL STATUS!!Females 15 years and over!!Widowed",
    "MARITAL STATUS!!Females 15 years and over!!Divorced",
    "SCHOOL ENROLLMENT!!Population 3 years and over enrolled in school",
    "SCHOOL ENROLLMENT!!Population 3 years and over enrolled in school!!Nursery school preschool",
    "SCHOOL ENROLLMENT!!Population 3 years and over enrolled in school!!Kindergarten",
    "SCHOOL ENROLLMENT!!Population 3 years and over enrolled in school!!Elementary school (grades 1-8)",
    "SCHOOL ENROLLMENT!!Population 3 years and over enrolled in school!!High school (grades 9-12)",
    "SCHOOL ENROLLMENT!!Population 3 years and over enrolled in school!!College or graduate school",
    "VETERAN STATUS!!Civilian population 18 years and over",
    "VETERAN STATUS!!Civilian population 18 years and over!!Civilian veterans",
    "DISABILITY STATUS OF THE CIVILIAN NONINSTITUTIONALIZED POPULATION!!Total Civilian Noninstitutionalized Population",
    "DISABILITY STATUS OF THE CIVILIAN NONINSTITUTIONALIZED POPULATION!!Total Civilian Noninstitutionalized Population!!With a disability",
    "DISABILITY STATUS OF THE CIVILIAN NONINSTITUTIONALIZED POPULATION!!Under 18 years",
    "DISABILITY STATUS OF THE CIVILIAN NONINSTITUTIONALIZED POPULATION!!Under 18 years!!With a disability",
    "DISABILITY STATUS OF THE CIVILIAN NONINSTITUTIONALIZED POPULATION!!18 to 64 years",
    "DISABILITY STATUS OF THE CIVILIAN NONINSTITUTIONALIZED POPULATION!!18 to 64 years!!With a disability",
    "DISABILITY STATUS OF THE CIVILIAN NONINSTITUTIONALIZED POPULATION!!65 years and over",
    "DISABILITY STATUS OF THE CIVILIAN NONINSTITUTIONALIZED POPULATION!!65 years and over!!With a disability",
    "TOT_MALE",
    "TOT_FEMALE",
    "WA_MALE",
    "WA_FEMALE",
    "BA_MALE",
    "BA_FEMALE",
    "IA_MALE",
    "IA_FEMALE",
    "AA_MALE",
    "AA_FEMALE",
    "NA_MALE",
    "NA_FEMALE",
    "TOM_MALE",
    "TOM_FEMALE",
    "WAC_MALE",
    "WAC_FEMALE",
    "BAC_MALE",
    "BAC_FEMALE",
    "IAC_MALE",
    "IAC_FEMALE",
    "AAC_MALE",
    "AAC_FEMALE",
    "NAC_MALE",
    "NAC_FEMALE",
    "NH_MALE",
    "NH_FEMALE",
    "NHWA_MALE",
    "NHWA_FEMALE",
    "NHBA_MALE",
    "NHBA_FEMALE",
    "NHIA_MALE",
    "NHIA_FEMALE",
    "NHAA_MALE",
    "NHAA_FEMALE",
    "NHNA_MALE",
    "NHNA_FEMALE",
    "NHTOM_MALE",
    "NHTOM_FEMALE",
    "NHWAC_MALE",
    "NHWAC_FEMALE",
    "NHBAC_MALE",
    "NHBAC_FEMALE",
    "NHIAC_MALE",
    "NHIAC_FEMALE",
    "NHAAC_MALE",
    "NHAAC_FEMALE",
    "NHNAC_MALE",
    "NHNAC_FEMALE",
    "H_MALE",
    "H_FEMALE",
    "HWA_MALE",
    "HWA_FEMALE",
    "HBA_MALE",
    "HBA_FEMALE",
    "HIA_MALE",
    "HIA_FEMALE",
    "HAA_MALE",
    "HAA_FEMALE",
    "HNA_MALE",
    "HNA_FEMALE",
    "HTOM_MALE",
    "HTOM_FEMALE",
    "HWAC_MALE",
    "HWAC_FEMALE",
    "HBAC_MALE",
    "HBAC_FEMALE",
    "HIAC_MALE",
    "HIAC_FEMALE",
    "HAAC_MALE",
    "HAAC_FEMALE",
    "HNAC_MALE",
    "HNAC_FEMALE",
    # "Active Physicians per 100000 Population 2018 (AAMC)",
    # "Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)",
    # "Active Primary Care Physicians per 100000 Population 2018 (AAMC)",
    # "Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)",
    # "Active General Surgeons per 100000 Population 2018 (AAMC)",
    # "Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)",
    # "Fraction of Active Physicians Who Are Female 2018 (AAMC)",
    # "Fraction of Active Physicians Who Are International Medical Graduates (IMGs) 2018 (AAMC)",
    # "Fraction of Active Physicians Who Are Age 60 or Older 2018 (AAMC)",
    # "MD and DO Student Enrollment per 100000 Population AY 2018-2019 (AAMC)",
    # "Student Enrollment at Public MD and DO Schools per 100000 Population AY 2018-2019 (AAMC)",
    # "Fraction Change in Student Enrollment at MD and DO Schools 2008-2018 (AAMC)",
    # "Fraction of MD Students Matriculating In-State AY 2018-2019 (AAMC)",
    # "Total Residents/Fellows in ACGME Programs per 100000 Population as of December 31 2018 (AAMC)",
    # "Total Residents/Fellows in Primary Care ACGME Programs per 100000 Population as of Dec. 31 2018 (AAMC)",
    # "Fraction of Residents in ACGME Programs Who Are IMGs as of December 31 2018 (AAMC)",
    # "Ratio of Residents and Fellows (GME) to Medical Students (UME) AY 2017-2018 (AAMC)",
    # "Percent Change in Residents and Fellows in ACGME-Accredited Programs 2008-2018 (AAMC)",
    # "Fraction of Physicians Retained in State from Undergraduate Medical Education (UME) 2018 (AAMC)",
    # "All Specialties (AAMC)",
    # "Allergy & Immunology (AAMC)",
    # "Anatomic/Clinical Pathology (AAMC)",
    # "Anesthesiology (AAMC)",
    # "Cardiovascular Disease (AAMC)",
    # "Child & Adolescent Psychiatry** (AAMC)",
    # "Critical Care Medicine (AAMC)",
    # "Dermatology (AAMC)",
    # "Emergency Medicine (AAMC)",
    # "Endocrinology Diabetes & Metabolism (AAMC)",
    # "Family Medicine/General Practice (AAMC)",
    # "Gastroenterology (AAMC)",
    # "General Surgery (AAMC)",
    # "Geriatric Medicine*** (AAMC)",
    # "Hematology & Oncology (AAMC)",
    # "Infectious Disease (AAMC)",
    # "Internal Medicine (AAMC)",
    # "Internal Medicine/Pediatrics (AAMC)",
    # "Interventional Cardiology (AAMC)",
    # "Neonatal-Perinatal Medicine (AAMC)",
    # "Nephrology (AAMC)",
    # "Neurological Surgery (AAMC)",
    # "Neurology (AAMC)",
    # "Neuroradiology (AAMC)",
    # "Obstetrics & Gynecology (AAMC)",
    # "Ophthalmology (AAMC)",
    # "Orthopedic Surgery (AAMC)",
    # "Otolaryngology (AAMC)",
    # "Pain Medicine & Pain Management (AAMC)",
    # "Pediatrics** (AAMC)",
    # "Physical Medicine & Rehabilitation (AAMC)",
    # "Plastic Surgery (AAMC)",
    # "Preventive Medicine (AAMC)",
    # "Psychiatry (AAMC)",
    # "Pulmonary Disease (AAMC)",
    # "Radiation Oncology (AAMC)",
    # "Radiology & Diagnostic Radiology (AAMC)",
    # "Rheumatology (AAMC)",
    # "Sports Medicine (AAMC)",
    # "Thoracic Surgery (AAMC)",
    # "Urology (AAMC)",
    # "Vascular & Interventional Radiology (AAMC)",
    # "Vascular Surgery (AAMC)",
    # "Total nurse practitioners (2019)",
    # "Total physician assistants (2019)",
    # "Total Hospitals (2019)",
    # "Internal Medicine Primary Care (2019)",
    # "Family Medicine/General Practice Primary Care (2019)",
    # "Pediatrics Primary Care (2019)",
    # "Obstetrics & Gynecology Primary Care (2019)",
    # "Geriatrics Primary Care (2019)",
    # "Total Primary Care Physicians (2019)",
    # "Psychiatry specialists (2019)",
    # "Surgery specialists (2019)",
    # "Anesthesiology specialists (2019)",
    # "Emergency Medicine specialists (2019)",
    # "Radiology specialists (2019)",
    # "Cardiology specialists (2019)",
    # "Oncology (Cancer) specialists (2019)",
    # "Endocrinology Diabetes and Metabolism specialists (2019)",
    # "All Other Specialties specialists (2019)",
    # "Total Specialist Physicians (2019)",
    "ICU Beds",
    # "transit_scores - population weighted averages aggregated from town/city level to county",
    "crime_rate_per_100000",
    "COUNTY POPULATION-AGENCIES REPORT ARRESTS",
    "COUNTY POPULATION-AGENCIES REPORT CRIMES",
    # "NUMBER OF AGENCIES IN COUNTY REPORT ARRESTS",
    # "NUMBER OF AGENCIES IN COUNTY REPORT CRIMES",
    "COVERAGE INDICATOR",
    "Total number of UCR (Uniform Crime Report) Index crimes excluding arson.",
    "Total number of UCR (Uniform Crime Report) index crimes reported including arson",
    "MURDER",
    "RAPE",
    "ROBBERY",
    "Number of AGGRAVATED ASSAULTS",
    "BURGLRY",
    "LARCENY",
    "MOTOR VEHICLE THEFTS",
    "ARSON"
  }

  intervention_keys = [
    'stay at home',
    '>50 gatherings',
    '>500 gatherings',
    'public schools',
    'restaurant dine-in',
    'entertainment/gym']

  per_what = 10000
  threshold = 50

  def __init__(self):
    self.counties = pd.read_csv(join(self.data_dir, 'counties.csv'), converters=self.converters)
    self.interventions = pd.read_csv(join(self.data_dir, 'interventions.csv'), converters=self.converters)
    self.infections = self._load_timeseries('infections')
    self.deaths = self._load_timeseries('deaths')
    self.descriptions = pd.read_csv(join(self.data_dir, 'list_of_columns.csv'), dtype=str)
    self.availability = pd.read_csv(join(self.data_dir, 'availability.csv'))

    # get the gradient of the time series
    self.infections_gradient = self.get_gradient(self.infections)
    self.deaths_gradient = self.get_gradient(self.deaths)

    # remove non-counties:
    is_county = list(map(self._is_county, list(self.counties.loc[:, 'FIPS'])))
    self.counties = self.counties.iloc[is_county, :]
    is_county = list(map(self._is_county, list(self.infections.loc[:, 'FIPS'])))
    self.infections = self.infections.iloc[is_county, :]
    self.deaths = self.deaths.iloc[is_county, :]

    # define the county names list, same ordering as counties
    self.fips_codes = list(self.counties['FIPS'])
    county_names = dict(
      FIPS=self.counties['FIPS'],
      county_name=[f'{capwords(area_name)}, {state}'
                   for area_name, state in zip(self.counties['Area_Name'], self.counties['State'])])
    self.county_names = pd.DataFrame(county_names)
    self.fips_to_county_name = dict(zip(self.fips_codes, county_names['county_name']))
    self.fips_to_population = dict(zip(self.fips_codes, self.counties['POP_ESTIMATE_2018']))

    # define the daily infections data, same ordering as infections
    date_key = self.infections.keys()[-1]
    month, day, year = map(int, date_key.split('/'))
    self.daily_infections_date = dt.date(year, month, day)
    # key = self.daily_infections_date.isoformat() + f' total infections per {self.per_what:,d}'
    infections_per_capita = [row[-1] / self.fips_to_population[row['FIPS']] * self.per_what
                             for idx, row in self.infections.iterrows()]
    self.total_infections = pd.DataFrame(
      {'FIPS': self.infections['FIPS'],
       'county_name': [self.fips_to_county_name[fips] for fips in self.infections['FIPS']],
       'infections_per_capita': infections_per_capita,
       'infections': self.infections.iloc[:, -1]})

    # define the start dates for the Analysis mode
    nonzeros = [np.array(row[1:] >= self.threshold).nonzero()[0] for i, row in self.infections.iterrows()]
    self.infections_start_indices = [nonzero[0] if nonzero.size > 0 else -1 for nonzero in nonzeros]
    # self.infections_start_dates = [nonzero[0] if nonzero.size > 0 else -1 for nonzero in nonzeros]

    # figure out the annotations for each FIPS in self.infections
    dates = [dt.date(int('20' + y), int(m), int(d)) for m, d, y in map(lambda x: x.split('/'), self.infections.keys()[1:])]
    self.timeseries_dates = [d.isoformat() for d in dates]
    timeseries_ordinal_dates = [d.toordinal() for d in dates]
    infections_start_dates_ordinal = [timeseries_ordinal_dates[i] for i in self.infections_start_indices]
    fips_to_interventions_row = dict((row[0], row) for i, row in self.interventions.iterrows())

    self.timeseries_start_index = (np.array(self.infections.iloc[:, 1:]) > 50).any(axis=0).nonzero()[0][0]

    # make annotations for the selected intervention on the graphs
    self.infections_annotations = {}       # (fips, intervention) -> annotation dict
    self.deaths_annotations = {}       # (fips, intervention key) -> annotation dict
    self.threshold_infections_annotations = {}       # (fips, intervention) -> annotation dict
    self.threshold_deaths_annotations = {}       # (fips, intervention key) -> annotation dict
    for i, row in self.infections.iterrows():
      fips = row['FIPS']
      interventions = fips_to_interventions_row[fips]
      for k in self.intervention_keys:
        if np.isnan(interventions[k]):
          continue
        d = dt.date.fromordinal(int(interventions[k]))
        d_idx_raw = int(interventions[k]) - timeseries_ordinal_dates[0]

        kwargs = dict(
          xref='x',
          yref='y',
          text=d.strftime('%b %d'),
          showarrow=True,
          arrowhead=0,
          ax=0,
          ay=-30,
          textangle=-90)

        self.infections_annotations[fips, k] = dict(
          x=d.isoformat(),
          xidx=d_idx_raw + 1,
          y=row[d_idx_raw + 1],
          **kwargs)
        
        self.deaths_annotations[fips, k] = dict(
          x=d.isoformat(),
          xidx=d_idx_raw + 1,
          y=self.deaths.iloc[i, d_idx_raw + 1],
          **kwargs)

        date_idx = d_idx_raw - self.infections_start_indices[i]
        if date_idx < 0:
          continue
        
        self.threshold_infections_annotations[fips, k] = dict(
          x=d_idx_raw - self.infections_start_indices[i],
          xidx=d_idx_raw + 1,
          y=row[d_idx_raw + 1] / self.fips_to_population[row[0]] * self.per_what,
          **kwargs)
        
        self.threshold_deaths_annotations[fips, k] = dict(
          x=d_idx_raw - self.infections_start_indices[i],
          xidx=d_idx_raw + 1,
          y=self.deaths.iloc[i, d_idx_raw + 1] / self.fips_to_population[row[0]] * self.per_what,
          **kwargs)

    # self.selected_county = list(self.infections.nlargest(1, date_key)['FIPS'])[0]
    self.selected_county = '53033'

    self._set_embedding()

    self.selected_counties = self.counties_subset_names['FIPS'][
      self.cluster_labels == self.fips_to_cluster_label[self.selected_county]]

  def set_selected_county(self, fips):
    if self.selected_county == fips:
      return
    self.selected_county = fips
    self.selected_cluster = self.fips_to_cluster_label[self.selected_county]
    self.selected_counties = self.counties_subset_names['FIPS'][
      self.cluster_labels == self.selected_cluster]
    
  def _is_county(self, fips):
    return fips[2:] != '000'

  def get_gradient(self, timeseries):
    # get the gradient of the time series
    gradient = np.array([savgol_filter(row[1:], window_length=7, polyorder=3, deriv=1) for i, row in timeseries.iterrows()])
    gradient = pd.DataFrame(gradient, columns=timeseries.keys()[1:])
    gradient = pd.concat([timeseries['FIPS'], gradient], axis=1)
    return gradient
    
  def get_counties_subset(self, selected_features=None):
    """Get the subset of counties with 100% availability for the selected features

    :returns: (numpy array of counties, df of identifiers for those counties)
    :rtype: 

    """
    if selected_features is None:
      selected_features = self.selected_features

    counties = self.counties.loc[:, selected_features]
    num_counties = counties.shape[0]
    which_counties = counties.notnull().values.all(axis=1)
    self.counties_subset_names = self.county_names.loc[which_counties]
    self.counties_subset = np.array(counties.iloc[which_counties, :])

    print(f'selected {self.counties_subset.shape[0]} / {num_counties} counties for embedding')
    return self.counties_subset, self.counties_subset_names

  reducer = umap.UMAP(n_neighbors=3, min_dist=0.03)  # metric=manhattan?
  # clusterer = DBSCAN(eps=0.3, min_samples=5, n_jobs=-1)
  clusterer = GaussianMixture(n_components=5)
  # clusterer = AgglomerativeClustering(n_clusters=5)  # 650?
  color_palette = sns.color_palette('hls', 10)
  color_palette = [f'#{int(255*t[0]):02x}{int(255*t[1]):02x}{int(255*t[2]):02x}' for t in color_palette]
  output_dir = 'output'
  if not exists(output_dir):
    os.mkdir(output_dir)

  selected_features = [
    # "POP_ESTIMATE_2018",
    # "N_POP_CHG_2018",
    # "NATURAL_INC_2018",
    "Some college or associate's degree 2014-18",
    "POVALL_2018",
    "Unemployed_2018",
    "Median_Household_Income_2018",
    "Housing units",
    "Male_age0to17",
    "Female_age0to17",
    "Male_age18to64",
    "Female_age18to64",
    "Male_age65plus",
    "Female_age65plus",
    "Area in square miles - Land area",
    "Housing units",
    "Density per square mile of land area - Population",    
    # "Density per square mile of land area - Housing units",
    # "ICU Beds",
    # "crime_rate_per_100000",
    # "COVERAGE INDICATOR",
    "transit_scores - population weighted averages aggregated from town/city level to county",
  ]

  features_to_normalize = {
    "N_POP_CHG_2018",
    "NATURAL_INC_2018",
    "Some college or associate's degree 2014-18",
    "POVALL_2018",
    "Unemployed_2018",
    "Housing units",
    "Male_age0to17",
    "Female_age0to17",
    "Male_age18to64",
    "Female_age18to64",
    "Male_age65plus",
    "Female_age65plus",
    "Housing units",
    "ICU Beds",
  }
    
  def _embed(self, x, fips_codes):
    print('FOR FAST DEBUGGING ONLY')
    fname = join(self.output_dir, 'embedding.npy')

    # normalize each column to zero mean, unit variance
    for j, feature in enumerate(self.selected_features):
      if feature in self.features_to_normalize:
        for i, fips in enumerate(fips_codes):
          x[i, j] /= self.fips_to_population[fips]

    x = (x - x.mean(axis=0, keepdims=True)) / np.sqrt(x.var(axis=0, keepdims=True) + 0.0001)
    
    if exists(fname) and False:
      embedding = np.load(fname)
    else:
      print('embedding...')
      embedding = self.reducer.fit_transform(x)
      np.save(fname, embedding)
      pd.DataFrame(dict(FIPS=fips_codes, x=embedding[:, 0], y=embedding[:, 1])).to_csv(join(self.output_dir, 'embedding.csv'))

    # self._plot_features(embedding, fips_codes)
      
    return embedding

  def _cluster(self, x, fips_codes):
    print('FOR FAST DEBUGGING ONLY')
    fname = join(self.output_dir, 'clustering.npy')
    if exists(fname) and False:
      labels = np.load(fname)
    else:
      print('clustering...')
      labels = self.clusterer.fit_predict(x)
      # labels = self.clusterer.labels_
      np.save(fname, labels)
      pd.DataFrame(dict(FIPS=fips_codes, x=x[:, 0], y=x[:, 1], cluster=labels)).to_csv(join(self.output_dir, 'clustering.csv'))
    return labels.astype(str)

  def _plot_features(self, x, fips_codes):
    counties = self.counties[self.counties['FIPS'].isin(fips_codes)]
    county_names = [self.fips_to_county_name[fips] for fips in fips_codes]
    
    for feature in self.selected_features:
      fig = go.Figure(go.Scatter(
        x=x[:, 0],
        y=x[:, 1],
        text=[f'{county_name}; {feature}: ' + (f'{value:,d}' if isinstance(value, int) else f'{value}')
              for county_name, value in zip(county_names, counties[feature])],
        hoverinfo='text+x+y',
        mode='markers',
        marker=dict(
          size=5,
          line={'width': 0.5, 'color': 'white'},
          color=np.log(counties[feature]),
          # colorscale='Magma',
          # showscale=False,
        )))
      fig.show()
  
  def _set_embedding(self, selected_features=None):
    counties_subset, counties_subset_names = self.get_counties_subset(selected_features=selected_features)
    subset_fips_codes = list(counties_subset_names['FIPS'])
    self.clustering_fips_codes = subset_fips_codes
    self.embedding = self._embed(counties_subset, subset_fips_codes)
    self.cluster_labels = self._cluster(counties_subset, subset_fips_codes)
    self.fips_to_cluster_label = dict(zip(counties_subset_names['FIPS'], self.cluster_labels))
    self.selected_cluster = self.fips_to_cluster_label[self.selected_county]
    self.unique_cluster_labels = set(self.cluster_labels)

    self.cluster_colors_map = dict(
      [('-1', '#000000')] + [(str(label), self.color_palette[int(label) % len(self.color_palette)])
                             for label in self.unique_cluster_labels if label != '-1'])
    self.cluster_colors = [self.cluster_colors_map[str(label)] for label in self.cluster_labels]
    self.clustering_df = pd.DataFrame(dict(
      FIPS=counties_subset_names['FIPS'],
      county_name=counties_subset_names['county_name'],
      cluster=self.cluster_labels.astype(str)))
    return self.embedding
    
  def _load_timeseries(self, timeseries_name):
    filename = join(self.data_dir, f'{timeseries_name}_timeseries.csv')
    timeseries = pd.read_csv(filename, converters=self.converters)
    timeseries = timeseries.drop(labels='Combined_Key', axis=1)
    return timeseries

  
if __name__ == '__main__':
  data = DashboardData()
  
  
