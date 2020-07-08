import os 
import wget
import argparse 
import datetime as dt
import numpy as np
import pandas as pd

from os.path import join, exists


class IHMEDataParser():
  """ Parses the rollback dates of the summary provided by IHME 
  """

  def __init__(self, data_dir):
    assert exists(data_dir), f'Check Data dir'
    self.data_dir = data_dir
    self.path = join(data_dir, 'us_data', 'IHME', 'Summary_stats_all_locs.csv')
    assert exists(self.path), f'Summary not available'

    self.interventions_path = join(self.data_dir, 'us_data', 'interventions_initial_submission.csv')
    self.fips_lookup_path = join(self.data_dir, 'us_data', 'FIPS_lookup.csv')
    self.parse(self.path, self.interventions_path, self.fips_lookup_path)

  def parse(self, ihme_path, interventions_path, fips_path):
    ihme_df = pd.read_csv(ihme_path, engine='python')
    int_df = pd.read_csv(interventions_path, engine='python', dtype={'FIPS': str})
    fips_df = pd.read_csv(fips_path, engine='python')

    # get list of states of the US
    states = fips_df['State'].unique().tolist()

    discard = ['D.C.', 'American Samoa', 'Guam', 'Northern Mariana Islands', 'U.S. Virgin Islands', 'Virgin Islands',
               'United States Virgin Islands', np.nan]
    for item in discard:
      states.remove(item)

    print(states)
    print(f'Number of states in the US: {len(states)}')

    # get the states' fips
    # fips = 

    # select all states of the US in IHME
    ihme_df = ihme_df[ihme_df['location_name'].isin(states)]

    # relevant interventions
    interventions = [  # 'travel_limit_end_date',
        'stay_home_end_date',
        # 'educational_fac_end_date',
        'any_gathering_restrict_end_date',
        # 'any_business_end_date',
        'all_non-ess_business_end_date'
    ]
    ihme_df = ihme_df[['location_name'] + interventions]

    # matching of interventions:
    # interventions_file | IHME 
    # ----------------------------------------
    # stay at home       | stay_home_end_date
    # public schools     | 
    # > 50 gathering     | any_gathering_restrict_end_date
    # > 500 gathering    | any_gathering_restrict_end_date
    # restaurant dine-in | all_non-ess_business_end_date
    # entertainment/gym  | all_non-ess_business_end_date
    # federal guideline  |
    # foreign travel ban |

    # move puerto rico to bottom
    pr = ihme_df[ihme_df['location_name'] == 'Puerto Rico'].copy()
    ihme_df = ihme_df.drop(253)  # index of pr
    ihme_df = ihme_df.append(pr, ignore_index=True)

    # fips state list
    fips_state_list = list(np.linspace(1000, 56000, 56, dtype=np.int))
    # fips_state_id_list = 
    fips_state_list += [72000]  # puerto rico is special, why can't they vote btw

    fips_series = pd.Series(fips_state_list, name='FIPS')
    states_to_fips_dict = dict(zip(ihme_df['location_name'].values.tolist(), fips_state_list))
    states_to_fips_dict = dict(zip(ihme_df['location_name'].values.tolist(), fips_state_list))
    ihme_df = pd.concat([fips_series, ihme_df], axis=1)

    # Define a dictionary with key values of 
    # an existing column and their respective 
    # value pairs as the # values for our new column. 

    columns_list = ['stay at home rollback',
                    '>50 gatherings rollback',
                    '>500 gatherings rollback',
                    'restaurant dine-in rollback',
                    'entertainment/gym rollback']

    merge_df = pd.DataFrame(columns=columns_list)
    c = columns_list
    x = interventions
    interventions_list = [x[0], x[1], x[1], x[2], x[2]]
    interventions_to_columns_dict = {x[0]: c[0],
                                     x[1]: [c[1], c[2]],
                                     x[2]: [c[3], c[4]]}

    # interventions_to_columns_dict = dict(zip(interventions_list, columns_list))

    print(interventions_to_columns_dict)
    int_df = pd.concat([int_df, merge_df], axis=1)

    for i in range(len(int_df)):
      if i == 0:
        continue
      fips_current = int_df.iloc[i, 0]
      id_ = int(fips_current[:2] + '000')

      date_current_list = ihme_df[ihme_df['FIPS'] == id_][interventions]
      date_current_list = date_current_list.values.tolist()

      for j in range(len(interventions)):
        date_current = date_current_list[0][j]
        if date_current != [] and type(date_current) != np.float:
          date_current = date_current

          # parse date
          date_current_ordinal = dt.dt.strptime(date_current, '%Y-%m-%d').toordinal()

          int_df.at[i, interventions_to_columns_dict[interventions[j]]] = date_current_ordinal

        else:
          int_df.at[i, interventions_to_columns_dict[interventions[j]]] = np.nan

    int_df = self.dirty_helper(int_df, columns_list)
    int_df.to_csv(join(self.data_dir, 'us_data', 'interventions_updated.csv'), index=True)

    # TODO: Check for 74000 and set the rollback accordingly

  def dirty_helper(self, df, columns_list):
    """ Hardcoded county rollback dates """
    print(df)

    df.set_index('FIPS', inplace=True)
    print('--------------BEFORE--------------')
    # print(df)
    columns_list = ['stay at home rollback',
                    '>50 gatherings rollback',
                    '>500 gatherings rollback',
                    'restaurant dine-in rollback',
                    'entertainment/gym rollback']
    c = columns_list
    # Alabama
    df.loc[df.STATE == 'AL', c[-5]] = dt.date(2020, 4, 30).toordinal()
    df.loc[df.STATE == 'AL', c[-1]] = dt.date(2020, 5, 11).toordinal()
    df.loc[df.STATE == 'AL', c[-2]] = dt.date(2020, 5, 11).toordinal()
    # Alaska
    # https://covid19.alaska.gov/wp-content/uploads/2020/05/05142020-Reopen-Alaska-Plan.pdf
    df.loc[df.STATE == 'AK', c[-5]] = dt.date(2020, 4, 24).toordinal()
    df.loc[df.STATE == 'AK', c[-4]] = dt.date(2020, 5, 8).toordinal()
    df.loc[df.STATE == 'AK', c[-3:]] = dt.date(2020, 5, 22).toordinal()
    # Arizona
    df.loc[df.STATE == 'AZ', c[-5]] = dt.date(2020, 5, 15).toordinal()

    # Arkansas
    df.loc[df.STATE == 'AR', c[-1]] = dt.date(2020, 5, 11).toordinal()
    df.loc[df.STATE == 'AR', c[-2]] = dt.date(2020, 5, 4).toordinal()

    # California
    # https://abc7news.com/sonoma-county-restaurants-napa-newsoms-plan-to-reopen-california-when/6178845/
    df.loc[['06003',  # Alpine
            '06005',  # Amador
            '06007',  # Butte
            '06009',  # Calaveras http://www.calaverasenterprise.com/news/coronavirus_information/article_da41af68-9577-11ea-affd-fb8bc483a30b.html
            '06011',  # Colusa
            '06015',  # Del Norte
            '06017',  # El Dorado
            '06019',  # Fresno
            '06021',  # Glenn
            '06023',  # Humboldt
            '06027',  # Inyo
            '06029',  # Kern
            '06031',  # Kings
            '06033',  # Lake
            '06035',  # Lassen
            '06039',  # Madera
            '06043',  # Mariposa
            '06045',  # Mendocino
            '06047',  # Merced
            '06049',  # Modoc
            '06051',  # Mono
            '06055',  # Napa
            '06057',  # Nevada
            '06059',  # Orange
            '06061',  # Placer
            '06063',  # Plumas
            '06065',  # Riverside
            '06067',  # Sacramento
            '06069',  # San Benito
            '06071',  # San Bernardino
            '06073',  # San Diego
            '06077',  # San Joaquin
            '06079',  # San Luis Obispo
            '06083',  # Santa Barbara
            '06089',  # Shasta
            '06091',  # Sierra
            '06093',  # Siskiyou
            '06095',  # Solano
            '06097',  # Sonoma
            '06099',  # Stanislaus
            '06103',  # Tehama
            '06105',  # Trinity
            '06109',  # Tuolumne
            '06111',  # Ventura
            '06113',  # Yolo
            '06115'   # Yuba
            ], [c[0], c[1], c[3], c[4]]] = dt.date(2020, 5, 13).toordinal()

    df.loc['06001', c[3]] = dt.date(2020, 6, 18).toordinal()  # Alameda http://www.acphd.org/2019-ncov/shelter-in-place.aspx
    df.loc['06053', [c[2], c[3], c[4]]] = dt.date(2020, 60, 12).toordinal()  # Monterey https://abc7news.com/reopening-california-monterey-tourism-shelter-in-place-carmel-by-the-sea/6257886/
    # LA county still in lockdown, reopening pushed back https://www.latimes.com/california/story/2020-05-12/coronavirus-beaches-reopen-los-angeles-county-move-toward-new-normal
    df.loc['06037', [c[3], c[4]]] = dt.date(2020, 6, 18).toordinal()  # https://covid19.lacounty.gov/covid19-news/nail-salons-spas-tattoo-shops-casinos-bars-and-wineries-reopening-with-safeguards/

    # https://sf.gov/step-by-step/reopening-san-francisco
    df.loc['06075', [c[0], c[3]]] = dt.date(2020, 6, 12).toordinal()  # other rollbacks have been pushed

    # San Mateo: https://www.smcgov.org/smc-reopening
    df.loc['06081', c[0]] = dt.date(2020, 5, 13).toordinal()
    df.loc['06081', [c[1], c[3], c[4]]] = dt.date(2020, 6, 17).toordinal()

    # Santa Cruz https://abc7news.com/santa-cruz-beaches-reopen-covid-is-open-coronavirus/6243867/
    df.loc['06087', [c[0], c[1], c[3], c[4]]] = dt.date(2020, 6, 12).toordinal()

    # Tulare https://www.visaliatimesdelta.com/story/news/2020/05/27/tulare-county-meets-readiness-criteria-can-reopen-pending-state-approval/5263823002/
    df.loc['06107', [c[0], c[1], c[3], c[4]]] = dt.date(2020, 5, 18).toordinal()

    # ====== end California =======

    # Colorado
    df.loc[df.STATE == 'CO', c[-5]] = dt.date(2020, 4, 26).toordinal()  # but stay at home again at July 1
    # Conneticut : https://portal.ct.gov/-/media/DECD/Covid_Business_Recovery-Phase-2/Amusement_Parks_C3_V1.pdf
    df.loc[df.STATE == 'CT', c[-5]] = dt.date(2020, 5, 20).toordinal()
    df.loc[df.STATE == 'CT', c[-2]] = dt.date(2020, 6, 17).toordinal()
    df.loc[df.STATE == 'CT', c[-1]] = dt.date(2020, 6, 17).toordinal()

    # Delaware
    #    restaurant and entertainment open from June 15th on src: https://coronavirus.delaware.gov/reopening/phase2/
    df.loc[df.STATE == 'DE', c[-1]] = dt.date(2020, 6, 15).toordinal()
    df.loc[df.STATE == 'DE', c[-2]] = dt.date(2020, 6, 15).toordinal()

    # DC 
    #   stay at home order June 22nd src: https://coronavirus.dc.gov/phasetwo
    df.loc[df.STATE == 'DC', c[-5]] = dt.date(2020, 6, 22).toordinal()

    # FL - checked for counties: phase 1 (restaurant and gym) all took effect at same time https://twitter.com/govrondesantis/status/1261369779035623425?lang=en
    #   stay at home order May 18 src: https://floridahealthcovid19.gov/plan-for-floridas-recovery/
    #   reastaurants May 18th 
    #   entertainment May 18th
    df.loc[df.STATE == 'FL', c[-5]] = dt.date(2020, 6, 18).toordinal()
    df.loc[df.STATE == 'FL', c[-2]] = dt.date(2020, 6, 18).toordinal()
    df.loc[df.STATE == 'FL', c[-1]] = dt.date(2020, 6, 18).toordinal()

    # Georgia : https://www.11alive.com/article/news/health/coronavirus/georgia-reopening-dates-plan-kemp/85-1df2aa97-48fd-4cf8-a9fd-afbd8c73dfcf
    df.loc[df.STATE == 'GA', c[-5]] = dt.date(2020, 4, 30).toordinal()
    df.loc[df.STATE == 'GA', c[-2]] = dt.date(2020, 4, 27).toordinal()
    df.loc[df.STATE == 'GA', c[-1]] = dt.date(2020, 4, 24).toordinal()
    # Hawaii
    df.loc[df.STATE == 'HI', c[-5]] = dt.date(2020, 5, 31).toordinal()

    # Idaho
    #   gatherings <50 May 30th src: https://rebound.idaho.gov/stages-of-reopening/
    #   gatherings <500 June 13th src: https://rebound.idaho.gov/stages-of-reopening/

    df.loc[df.STATE == 'ID', c[-5]] = dt.date(2020, 5, 1).toordinal()
    df.loc[df.STATE == 'ID', c[-4]] = dt.date(2020, 5, 30).toordinal()
    df.loc[df.STATE == 'ID', c[-3]] = dt.date(2020, 6, 13).toordinal()

    # Illinois
    # stay at home order May 30th src:https://www.pantagraph.com/news/state-and-regional/illinois-stay-at-home-order-ends-and-restrictions-lifted-on-churches-as-the-state-advances/article_71393207-40a5-58cf-a658-c580da3d437d.html
    df.loc[df.STATE == 'IL', c[-5]] = dt.date(2020, 5, 30).toordinal()
    # Indiana 
    # Iowa
    # https://wcfcourier.com/news/local/govt-and-politics/update-watch-now-iowa-to-reopen-restaurants-friday/article_7636be19-9dec-5cb9-8344-29c6aafd0196.html
    df.loc[df.STATE == 'IA', c[-1]] = dt.date(2020, 5, 11).toordinal()  # https://www.thegazette.com/subject/news/business/gyms-working-up-a-sweat-to-reopen-friday-20200514
    df.loc[df.FIPS == '19153', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19005', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19011', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19013', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19017', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19049', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19057', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19061', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19065', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19087', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19095', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19099', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19103', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19113', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19115', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19127', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19139', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19157', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19163', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19171', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19183', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '19193', c[-2:]] = dt.date(2020, 5, 15).toordinal()

    # Kansas 
    # Kentucky
    # Lousiana
    # indoor limit <250 people not considered: src: https://gov.louisiana.gov/index.cfm/newsroom/detail/2573
    # Maine
    # Maryland https://conduitstreet.mdcounties.org/2020/05/15/marylands-reopening-status-by-county/
    # Anne Arundel County https://www.aacounty.org/coronavirus/road-to-recovery/
    df.loc[df.FIPS == '24003', c[-2:]] = dt.date(2020, 5, 29).toordinal() 
    df.loc[df.FIPS == '24003', c[-1:]] = dt.date(2020, 6, 4).toordinal() 

    # Baltimore https://baltimore.cbslocal.com/reopening-maryland-whats-open-whats-closed-county-by-county/
    df.loc[df.FIPS == '24510', c[-2:]] = dt.date(2020, 5, 29).toordinal() 
    df.loc[df.FIPS == '24510', c[-1:]] = dt.date(2020, 6, 19).toordinal() 

    # Baltimore County https://www.baltimorecountymd.gov/News/BaltimoreCountyNow/baltimore-county-to-fully-enter-stage-one-reopening
    df.loc[df.FIPS == '24005', c[-2:]] = dt.date(2020, 5, 29).toordinal()
    df.loc[df.FIPS == '24005', c[-1:]] = dt.date(2020, 6, 5).toordinal()

    # Charles County https://www.charlescountymd.gov/services/health-and-human-services/covid-19
    df.loc[df.FIPS == '24005', c[-2:]] = dt.date(2020, 5, 29).toordinal()
    df.loc[df.FIPS == '24005', c[-1:]] = dt.date(2020, 6, 19).toordinal()

    # Frederick County https://health.frederickcountymd.gov/621/Recovery
    df.loc[df.FIPS == '24021', c[-2:]] = dt.date(2020, 5, 29).toordinal()
    df.loc[df.FIPS == '24021', c[-1:]] = dt.date(2020, 6, 19).toordinal()

    # Howard County https://www.howardcountymd.gov/News/ArticleID/2007/Coronavirus-Updates-Howard-County-Aligns-with-Governor%E2%80%99s-Phase-2-Reopening-Contact-Tracing-Campaign
    df.loc[df.FIPS == '24027', c[-2:]] = dt.date(2020, 5, 29).toordinal()
    df.loc[df.FIPS == '24027', c[-1:]] = dt.date(2020, 6, 5).toordinal()

    # Montgomery County https://www.montgomerycountymd.gov/covid19/news/index.html
    df.loc[df.FIPS == '24031', c[-2:]] = dt.date(2020, 6, 1).toordinal()
    df.loc[df.FIPS == '24031', c[-1:]] = dt.date(2020, 6, 19).toordinal()

    # Prince George County https://www.princegeorgescountymd.gov/Archive.aspx?AMID=142
    df.loc[df.FIPS == '24033', c[-2:]] = dt.date(2020, 6, 1).toordinal()
    df.loc[df.FIPS == '24033', c[-1:]] = dt.date(2020, 6, 29).toordinal()

    # Massachusetts - checked, seems to be all counties the same
    #   restaurants: June 22nd src: https://www.mass.gov/info-details/safety-standards-and-checklist-restaurants
    df.loc[df.STATE == 'MA', c[-2]] = dt.date(2020, 6, 22).toordinal()
    # Michigan MI
    # Minnesota MN
    #   restaurant  & entertainment June 10th src: https://mn.gov/covid19/for-minnesotans/stay-safe-mn/stay-safe-plan.jsp
    df.loc[df.STATE == 'MN', c[-2]] = dt.date(2020, 6, 10).toordinal()
    df.loc[df.STATE == 'MN', c[-1]] = dt.date(2020, 6, 10).toordinal()
    # Mississippi MS

    # Missouri MO # state wide lifted on June 16th, but local gov can impose rules src: https://governor.mo.gov/show-me-strong-recovery-plan-guidance-and-frequently-asked-questions
    #   stay at home ended May 4th: https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html
    #   all other June 15th src: https://www.sos.mo.gov/library/reference/orders/2020/eo12
    df.loc[df.STATE == 'MO', c[-5]] = dt.date(2020, 5, 4).toordinal()
    for i in range(1, 5):
      df.loc[df.STATE == 'MO', c[-i]] = dt.date(2020, 6, 15).toordinal()

    df.loc['29189', [c[1], c[2]]] = dt.date(2020, 6, 29).toordinal()  # st louis county https://stlpartnership.com/details-on-state-stl-county-and-city-of-stl-reopening/
    

    # Montana MT
    # Nebraska NA  
    # No stay at home order imposed, remove rollback src: https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html
    df.loc[df.STATE == 'NE', c[-5]] = np.nan
    # Nevada NV
    # New Hampshire NH
    #   stay at home June 15th src: https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html
    df.loc[df.STATE == 'NH', c[-5]] = dt.date(2020, 6, 15).toordinal()

    # New Jersey NJ - appears to have opened at once https://www.thegazette.com/subject/news/business/gyms-working-up-a-sweat-to-reopen-friday-20200514
    #   no indoor dining - but outdoor
    df.loc[df.STATE == 'NJ', c[-2]] = dt.date(2020, 6, 15).toordinal()  # np.nan
    df.loc[df.STATE == 'NJ', c[-1]] = dt.date(2020, 7, 2).toordinal()  # np.nan

    # New Mexico NM - apparently not actually what IHME said? https://www.newmexico.gov/2020/05/28/governor-announces-limited-reopening-for-dine-in-restaurants-indoor-malls-gyms-salons-and-more/

    df.loc[df.STATE == 'NM', c[-2]] = dt.date(2020, 5, 28)
    df.loc[df.STATE == 'NM', c[-1]] = dt.date(2020, 6, 1)

    # Some counties were weird: https://www.usnews.com/news/best-states/new-mexico/articles/2020-05-25/new-mexico-governor-blocks-plans-to-reopen-drive-in-theater
    df.loc[df.FIPS == '35006', c[-2:]] = dt.date(2020, 6, 1).toordinal()  # Cibola
    df.loc[df.FIPS == '35031', c[-2:]] = dt.date(2020, 6, 1).toordinal()  # McKinley
    df.loc[df.FIPS == '35045', c[-2:]] = dt.date(2020, 6, 1).toordinal()  # San Juan

    # New York NY
    # Only using New York County for all 5 boroughs since that's where death is counted
    # Gyms are not included in reopening plan as of yet - https://nymag.com/intelligencer/2020/07/when-will-new-york-reopen-phases-and-full-plan-explained.html
    df.loc[df.STATE == 'NY', c[-1]] = np.nan

    # https://www1.nyc.gov/nycbusiness/article/nyc-restaurant-reopening-guide
    df.loc[df.FIPS == '36061', c[-2:]] = dt.date(2020, 6, 22).toordinal() 
    df.loc[df.FIPS == '36061', c[-2:]] = np.nan

    # Counties https://www.governor.ny.gov/news/governor-cuomo-announces-outdoor-dining-restaurants-will-be-permitted-phase-two-reopening
    # Capital Region - Albany, Columbia, Greene, Saratoga, Schenectady, Rensselaer, Warren, Washington
    df.loc[df.FIPS == '36001', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36021', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36039', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36091', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36093', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36083', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36113', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36115', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    # Central New York - Cayuga, Cortland, Madison, Onondaga, Oswego
    df.loc[df.FIPS == '36011', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36023', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36053', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36067', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36075', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    # the Finger Lakes - Genesee, Livingston, Monroe, Ontario, Orleans, Seneca, Wayne, Wyoming, Yates
    df.loc[df.FIPS == '36037', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36051', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36055', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36069', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36073', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36099', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36117', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36121', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36123', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    # the Mohawk Valley - Fulton, Herkimer, Montgomery, Oneida, Otsego, Schoharie
    df.loc[df.FIPS == '36035', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36043', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36057', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36065', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36077', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36095', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    # the North Country - Clinton, Essex, Franklin, Hamilton, Jefferson, Lewis, St. Lawrence
    df.loc[df.FIPS == '36019', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36031', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36033', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36041', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36045', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36049', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36089', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    # the Southern Tier Broome, Chemung, Chenango, Delaware, Schuyler, Steuben, Tioga, Tompkins
    df.loc[df.FIPS == '36007', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36015', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36017', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36025', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36097', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36101', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36107', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36109', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    # Western New York - Allegany, Cattaraugus, Chautauqua, Erie, Niagara
    df.loc[df.FIPS == '36003', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36009', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36013', c[-2:]] = dt.date(2020, 6, 4).toordinal()
    df.loc[df.FIPS == '36029', c[-2:]] = dt.date(2020, 6, 4).toordinal() 
    df.loc[df.FIPS == '36063', c[-2:]] = dt.date(2020, 6, 4).toordinal()

    # Long Island - https://www.nbcnewyork.com/news/local/hundreds-of-restaurants-expected-to-reopen-on-long-island-as-phase-ii-begins-cuomo-shifts-metrics-focus/2454500/
    df.loc[df.FIPS == '36059', c[-2:]] = dt.date(2020, 6, 10).toordinal() 
    df.loc[df.FIPS == '36103', c[-2:]] = dt.date(2020, 6, 10).toordinal()

    # Mid-Hudson https://www.dailyfreeman.com/news/local-news/phase-2-starts-tuesday-in-mid-hudson-region-outdoor-dining-at-restaurants-haircuts-in-store/article_afba1bea-a9a4-11ea-bf5a-677d5abe84e2.html
    # Dutchess, Orange, Putnam, Rockland, Sullivan, Ulster, Westchester
    df.loc[df.FIPS == '36027', c[-2:]] = dt.date(2020, 6, 9).toordinal() 
    df.loc[df.FIPS == '36071', c[-2:]] = dt.date(2020, 6, 9).toordinal()
    df.loc[df.FIPS == '36079', c[-2:]] = dt.date(2020, 6, 9).toordinal() 
    df.loc[df.FIPS == '36087', c[-2:]] = dt.date(2020, 6, 9).toordinal()
    df.loc[df.FIPS == '36105', c[-2:]] = dt.date(2020, 6, 9).toordinal() 
    df.loc[df.FIPS == '36111', c[-2:]] = dt.date(2020, 6, 9).toordinal() 
    df.loc[df.FIPS == '36119', c[-2:]] = dt.date(2020, 6, 9).toordinal()

    # Phase 4 includes gathering of 50 people
    # Capital Region - https://www.governor.ny.gov/news/governor-cuomo-announces-capital-region-cleared-global-public-health-experts-enter-phase-4
    df.loc[df.FIPS == '36001', c[-4:]] = dt.date(2020, 7, 1).toordinal() 
    df.loc[df.FIPS == '36021', c[-4:]] = dt.date(2020, 7, 1).toordinal()
    df.loc[df.FIPS == '36039', c[-4:]] = dt.date(2020, 7, 1).toordinal() 
    df.loc[df.FIPS == '36091', c[-4:]] = dt.date(2020, 7, 1).toordinal() 
    df.loc[df.FIPS == '36093', c[-4:]] = dt.date(2020, 7, 1).toordinal()
    df.loc[df.FIPS == '36083', c[-4:]] = dt.date(2020, 7, 1).toordinal() 
    df.loc[df.FIPS == '36113', c[-4:]] = dt.date(2020, 7, 1).toordinal() 
    df.loc[df.FIPS == '36115', c[-4:]] = dt.date(2020, 7, 1).toordinal()

    # https://www.governor.ny.gov/news/governor-cuomo-announces-five-regions-track-enter-phase-iv-reopening-friday
    # Central New York - Cayuga, Cortland, Madison, Onondaga, Oswego
    df.loc[df.FIPS == '36011', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36023', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36053', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36067', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36075', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    # the Finger Lakes - Genesee, Livingston, Monroe, Ontario, Orleans, Seneca, Wayne, Wyoming, Yates
    df.loc[df.FIPS == '36037', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36051', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36055', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36069', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36073', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36099', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36117', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36121', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36123', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    # the Mohawk Valley - Fulton, Herkimer, Montgomery, Oneida, Otsego, Schoharie
    df.loc[df.FIPS == '36035', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36043', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36057', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36065', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36077', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36095', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    # the North Country - Clinton, Essex, Franklin, Hamilton, Jefferson, Lewis, St. Lawrence
    df.loc[df.FIPS == '36019', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36031', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36033', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36041', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36045', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36049', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36089', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    # the Southern Tier Broome, Chemung, Chenango, Delaware, Schuyler, Steuben, Tioga, Tompkins
    df.loc[df.FIPS == '36007', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36015', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36017', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36025', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36097', c[-4:]] = dt.date(2020, 6, 26).toordinal() 
    df.loc[df.FIPS == '36101', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36107', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    df.loc[df.FIPS == '36109', c[-4:]] = dt.date(2020, 6, 26).toordinal()
    # Western New York - https://spectrumlocalnews.com/nys/buffalo/politics/2020/06/29/western-new-york-begins-phase-4-reopening-tuesday-
    df.loc[df.FIPS == '36003', c[-4:]] = dt.date(2020, 6, 30).toordinal()
    df.loc[df.FIPS == '36009', c[-4:]] = dt.date(2020, 6, 30).toordinal() 
    df.loc[df.FIPS == '36013', c[-4:]] = dt.date(2020, 6, 30).toordinal()
    df.loc[df.FIPS == '36029', c[-4:]] = dt.date(2020, 6, 30).toordinal() 
    df.loc[df.FIPS == '36063', c[-4:]] = dt.date(2020, 6, 30).toordinal()

    # https://www.dailyfreeman.com/news/local-news/mid-hudson-region-starts-phase-4-of-reopening-process-on-tuesday/article_bf23d59c-bf9f-11ea-bd04-979c464b1ebc.html
    # Long Island
    df.loc[df.FIPS == '36059', c[-4:]] = dt.date(2020, 7, 8).toordinal() 
    df.loc[df.FIPS == '36103', c[-4:]] = dt.date(2020, 7, 8).toordinal()

    # Mid-Hudson 
    # Dutchess, Orange, Putnam, Rockland, Sullivan, Ulster, Westchester
    df.loc[df.FIPS == '36027', c[-4:]] = dt.date(2020, 7, 8).toordinal() 
    df.loc[df.FIPS == '36071', c[-4:]] = dt.date(2020, 7, 8).toordinal()
    df.loc[df.FIPS == '36079', c[-4:]] = dt.date(2020, 7, 8).toordinal() 
    df.loc[df.FIPS == '36087', c[-4:]] = dt.date(2020, 7, 8).toordinal()
    df.loc[df.FIPS == '36105', c[-4:]] = dt.date(2020, 7, 8).toordinal() 
    df.loc[df.FIPS == '36111', c[-4:]] = dt.date(2020, 7, 8).toordinal() 
    df.loc[df.FIPS == '36119', c[-4:]] = dt.date(2020, 7, 8).toordinal()

    # North Carolina NC 
    #   no gym src: https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html
    # Avoided county-by-county approach - https://www.newsobserver.com/news/coronavirus/article242836711.html
    df.loc[df.STATE == 'NC', c[-1]] = np.nan

    # North Dakota ND - seems to reopen at once https://www.usnews.com/news/best-states/north-dakota/articles/2020-05-01/north-dakota-cafes-other-businesses-reopen-under-new-rules

    # Ohio OH
    #   entertainment/gyms June 10th src: https://coronavirus.ohio.gov/wps/portal/gov/covid-19/resources/news-releases-news-you-can-use/governor-reopen-certain-facilities
    #   restaurants dine in May 21st src: https://coronavirus.ohio.gov/wps/portal/gov/covid-19/resources/news-releases-news-you-can-use/reopening-restaurants-bars-personal-care-services
    df.loc[df.STATE == 'OH', c[-4]] = dt.date(2020, 5, 21).toordinal()
    df.loc[df.STATE == 'OH', c[-5]] = dt.date(2020, 6, 10).toordinal()
    # Oklahoma OK
    # Oregon OR
    # County response: https://govstatus.egov.com/reopening-oregon#countyStatuses
    df.loc[df.STATE == 'Or', c[-5]] = dt.date(2020, 7, 6).toordinal() 
    # Phase 1: Clackamas County, Lincoln County, Multnomah County, Washington County

    df.loc[df.FIPS == '41005', c[-2:]] = dt.date(2020, 5, 23).toordinal()  # clackamas https://www.clackamas.us/coronavirus/updates
    df.loc[df.FIPS == '41041', c[-2:]] = dt.date(2020, 5, 15).toordinal()  # lincoln
    df.loc[df.FIPS == '41051', c[-2:]] = dt.date(2020, 6, 19).toordinal()  # multnomah
    df.loc[df.FIPS == '41067', c[-2:]] = dt.date(2020, 6, 1).toordinal()  # washington

    df.loc[df.FIPS == '41047', c[-2:]] = dt.date(2020, 5, 22).toordinal()  # marion county https://www.co.marion.or.us/HLT/COVID-19/Pages/Reopening.aspx
    df.loc[df.FIPS == '41053', c[-2:]] = dt.date(2020, 5, 22).toordinal()  # polk https://www.co.polk.or.us/ph/covid-19-news

    # Phase 2: Rest
    # https://www.oregon.gov/newsroom/Pages/NewsDetail.aspx?newsid=36806
    # https://www.kgw.com/article/news/health/coronavirus/these-oregon-counties-have-been-approved-for-phase-1-of-reopening/283-b24c4243-bb25-43e7-bafa-75e5126a71a0

    df.loc[df.FIPS == '41001', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41003', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41007', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41009', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41011', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41013', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41015', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41017', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41019', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41021', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41023', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41025', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41027', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41029', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41031', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41033', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41035', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41037', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41039', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41043', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41045', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41049', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41055', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41057', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41059', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41061', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41063', c[-2:]] = dt.date(2020, 5, 15).toordinal()
    df.loc[df.FIPS == '41065', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41069', c[-2:]] = dt.date(2020, 5, 15).toordinal() 
    df.loc[df.FIPS == '41071', c[-2:]] = dt.date(2020, 5, 15).toordinal()

    # Pennsylvania PA
    # county response :
    # yellow phase: https://www.wgal.com/article/pennsylvania-counties-reopening-coronavirus/32343176#
    # https://www.governor.pa.gov/newsroom/gov-wolf-12-more-counties-to-move-to-yellow-phase-on-may-22/
    # https://www.pahomepage.com/news/wolf-announces-next-counties-to-move-to-yellow-phase/: As of today, the following counties are expected to ease restrictions on the ‘stay at home’ order on May 15, moving them from the ‘red phase’ to the yellow phase.’ Those counties include Allegheny, Armstrong, Bedford, Blair, Butler, Cambria, Fayette, Fulton, Greene, Indiana, Somerset, Washington and Westmoreland.
    # Additionally, eight counties will be able to move to yellow on May 29: Dauphin, Franklin, Huntingdon, Lebanon, Luzerne, Monroe, Pike, and Schuylkill.
    # The last to go will be Berks, Bucks, Chester, Delaware, Lackawanna, Lancaster, Lehigh, Northampton, Montgomery, and Philadelphia - June 5
    df.loc[df.FIPS == '42001', c[-5]] = dt.date(2020, 5, 22).toordinal()  # adams
    df.loc[df.FIPS == '42003', c[-5]] = dt.date(2020, 5, 15).toordinal()  # allegheny
    df.loc[df.FIPS == '42005', c[-5]] = dt.date(2020, 5, 15).toordinal()  # armstrong
    df.loc[df.FIPS == '42007', c[-5]] = dt.date(2020, 5, 22).toordinal()  # beaver
    df.loc[df.FIPS == '42009', c[-5]] = dt.date(2020, 5, 15).toordinal()  # bedford
    df.loc[df.FIPS == '42011', c[-5]] = dt.date(2020, 6, 5).toordinal()  # berks
    df.loc[df.FIPS == '42013', c[-5]] = dt.date(2020, 5, 15).toordinal()  # blair
    df.loc[df.FIPS == '42015', c[-5]] = dt.date(2020, 5, 8).toordinal()  # bradford
    df.loc[df.FIPS == '42017', c[-5]] = dt.date(2020, 6, 5).toordinal()  # bucks
    df.loc[df.FIPS == '42019', c[-5]] = dt.date(2020, 5, 15).toordinal()  # butler
    df.loc[df.FIPS == '42021', c[-5]] = dt.date(2020, 5, 15).toordinal()  # cambria
    df.loc[df.FIPS == '42023', c[-5]] = dt.date(2020, 5, 8).toordinal()  # cameron
    df.loc[df.FIPS == '42025', c[-5]] = dt.date(2020, 5, 22).toordinal()  # carbon
    df.loc[df.FIPS == '42027', c[-5]] = dt.date(2020, 5, 8).toordinal()  # centre
    df.loc[df.FIPS == '42029', c[-5]] = dt.date(2020, 6, 5).toordinal()  # chester
    df.loc[df.FIPS == '42031', c[-5]] = dt.date(2020, 5, 8).toordinal()  # clarion
    df.loc[df.FIPS == '42033', c[-5]] = dt.date(2020, 5, 8).toordinal()  # clearfied
    df.loc[df.FIPS == '42035', c[-5]] = dt.date(2020, 5, 8).toordinal()  # clinton
    df.loc[df.FIPS == '42037', c[-5]] = dt.date(2020, 5, 22).toordinal()  # columbia
    df.loc[df.FIPS == '42039', c[-5]] = dt.date(2020, 5, 8).toordinal()  # crawford
    df.loc[df.FIPS == '42041', c[-5]] = dt.date(2020, 5, 22).toordinal()  # cumberland
    df.loc[df.FIPS == '42043', c[-5]] = dt.date(2020, 5, 29).toordinal()  # dauphin
    df.loc[df.FIPS == '42045', c[-5]] = dt.date(2020, 6, 5).toordinal()  # delaware
    df.loc[df.FIPS == '42047', c[-5]] = dt.date(2020, 5, 8).toordinal()  # elk
    df.loc[df.FIPS == '42049', c[-5]] = dt.date(2020, 5, 8).toordinal()  # erie
    df.loc[df.FIPS == '42051', c[-5]] = dt.date(2020, 5, 15).toordinal()  # fayette
    df.loc[df.FIPS == '42053', c[-5]] = dt.date(2020, 5, 8).toordinal()  # forest
    df.loc[df.FIPS == '42055', c[-5]] = dt.date(2020, 5, 29).toordinal()  # franklin
    df.loc[df.FIPS == '42057', c[-5]] = dt.date(2020, 5, 15).toordinal()  # fulton
    df.loc[df.FIPS == '42059', c[-5]] = dt.date(2020, 5, 15).toordinal()  # greene
    df.loc[df.FIPS == '42061', c[-5]] = dt.date(2020, 5, 29).toordinal()  # huntingdon
    df.loc[df.FIPS == '42063', c[-5]] = dt.date(2020, 5, 15).toordinal()  # indiana
    df.loc[df.FIPS == '42065', c[-5]] = dt.date(2020, 5, 8).toordinal()  # jefferson
    df.loc[df.FIPS == '42067', c[-5]] = dt.date(2020, 5, 22).toordinal()  # juniata
    df.loc[df.FIPS == '42069', c[-5]] = dt.date(2020, 6, 5).toordinal()  # lackawannaa
    df.loc[df.FIPS == '42071', c[-5]] = dt.date(2020, 6, 5).toordinal()  # lancaster
    df.loc[df.FIPS == '42073', c[-5]] = dt.date(2020, 5, 8).toordinal()  # lawrence
    df.loc[df.FIPS == '42075', c[-5]] = dt.date(2020, 5, 29).toordinal()  # lebanon####
    df.loc[df.FIPS == '42077', c[-5]] = dt.date(2020, 6, 5).toordinal()  # lehigh
    df.loc[df.FIPS == '42079', c[-5]] = dt.date(2020, 5, 29).toordinal()  # luzerne
    df.loc[df.FIPS == '42081', c[-5]] = dt.date(2020, 5, 8).toordinal()  # lycoming
    df.loc[df.FIPS == '42083', c[-5]] = dt.date(2020, 5, 8).toordinal()  # mckean
    df.loc[df.FIPS == '42085', c[-5]] = dt.date(2020, 5, 8).toordinal()  # mercer
    df.loc[df.FIPS == '42087', c[-5]] = dt.date(2020, 5, 22).toordinal()  # mifflin
    df.loc[df.FIPS == '42089', c[-5]] = dt.date(2020, 5, 29).toordinal()  # monroe
    df.loc[df.FIPS == '42091', c[-5]] = dt.date(2020, 6, 5).toordinal()  # montgomery
    df.loc[df.FIPS == '42093', c[-5]] = dt.date(2020, 5, 8).toordinal()  # montour
    df.loc[df.FIPS == '42095', c[-5]] = dt.date(2020, 6, 5).toordinal()  # northampton
    df.loc[df.FIPS == '42097', c[-5]] = dt.date(2020, 5, 8).toordinal()  # northumberland
    df.loc[df.FIPS == '42099', c[-5]] = dt.date(2020, 5, 22).toordinal()  # perry
    df.loc[df.FIPS == '42101', c[-5]] = dt.date(2020, 6, 5).toordinal()  # philadelphia
    df.loc[df.FIPS == '42103', c[-5]] = dt.date(2020, 5, 29).toordinal()  # pike
    df.loc[df.FIPS == '42105', c[-5]] = dt.date(2020, 5, 8).toordinal()  # potter
    df.loc[df.FIPS == '42107', c[-5]] = dt.date(2020, 5, 29).toordinal()  # schuylkill
    df.loc[df.FIPS == '42109', c[-5]] = dt.date(2020, 5, 8).toordinal()  # snyder
    df.loc[df.FIPS == '42111', c[-5]] = dt.date(2020, 5, 15).toordinal()  # somerset
    df.loc[df.FIPS == '42113', c[-5]] = dt.date(2020, 5, 8).toordinal()  # sullivan
    df.loc[df.FIPS == '42115', c[-5]] = dt.date(2020, 5, 22).toordinal()  # susquehanna
    df.loc[df.FIPS == '42117', c[-5]] = dt.date(2020, 5, 8).toordinal()  # tioga
    df.loc[df.FIPS == '42119', c[-5]] = dt.date(2020, 5, 8).toordinal()  # union
    df.loc[df.FIPS == '42121', c[-5]] = dt.date(2020, 5, 8).toordinal()  # venango
    df.loc[df.FIPS == '42123', c[-5]] = dt.date(2020, 5, 8).toordinal()  # warren
    df.loc[df.FIPS == '42125', c[-5]] = dt.date(2020, 5, 15).toordinal()  # washington
    df.loc[df.FIPS == '42127', c[-5]] = dt.date(2020, 5, 22).toordinal()  # wayne  
    df.loc[df.FIPS == '42129', c[-5]] = dt.date(2020, 5, 15).toordinal()  # westmoreland
    df.loc[df.FIPS == '42131', c[-5]] = dt.date(2020, 5, 22).toordinal()  # wyoming
    df.loc[df.FIPS == '42133', c[-5]] = dt.date(2020, 5, 22).toordinal()  # york

    # green phase:	https://www.mcall.com/coronavirus/mc-nws-coronavirus-pa-counties-reopening-20200509-hqwbnzot5bb6tlw3g3j7qalxhq-story.html
    #   June 26: Berks, Bucks, Chester, Delaware, Erie, Lackawanna, Lancaster, Lehigh, Montgomery, Northampton, Philadelphia, Susquehanna;
    #	June 19: Dauphin, Franklin, Huntingdon, Luzerne, Monroe, Perry, Pike and Schuylkill.

    #	June 12: Adams, Beaver, Carbon, Columbia, Cumberland, Juniata, Mifflin, Northumberland, Union, Wayne, Wyoming and York.

    #	June 5: Allegheny, Armstrong, Bedford, Blair, Butler, Cambria, Clinton, Fayette, Fulton, Greene, Indiana, Lycoming, Mercer, Somerset, Washington and Westmoreland.

    #	May 29: These 18 counties became the first to move to the green phase, the third and final stage of the reopening plan: Bradford, Cameron, Centre, Clarion, Clearfield, Crawford, Elk, Forest, Jefferson, Lawrence, McKean, Montour, Potter, Snyder, Sullivan, Tioga, Venango and Warren counties.

    df.loc[df.FIPS == '42001', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # adams
    df.loc[df.FIPS == '42003', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # allegheny
    df.loc[df.FIPS == '42005', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # armstrong
    df.loc[df.FIPS == '42007', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # beaver
    df.loc[df.FIPS == '42009', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # bedford
    df.loc[df.FIPS == '42011', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # berks
    df.loc[df.FIPS == '42013', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # blair
    df.loc[df.FIPS == '42015', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # bradford
    df.loc[df.FIPS == '42017', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # bucks
    df.loc[df.FIPS == '42019', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # butler
    df.loc[df.FIPS == '42021', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # cambria
    df.loc[df.FIPS == '42023', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # cameron
    df.loc[df.FIPS == '42025', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # carbon
    df.loc[df.FIPS == '42027', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # centre
    df.loc[df.FIPS == '42029', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # chester
    df.loc[df.FIPS == '42031', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # clarion
    df.loc[df.FIPS == '42033', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # clearfied
    df.loc[df.FIPS == '42035', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # clinton
    df.loc[df.FIPS == '42037', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # columbia
    df.loc[df.FIPS == '42039', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # crawford
    df.loc[df.FIPS == '42041', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # cumberland
    df.loc[df.FIPS == '42043', c[-4:]] = dt.date(2020, 6, 19).toordinal()  # dauphin
    df.loc[df.FIPS == '42045', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # delaware
    df.loc[df.FIPS == '42047', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # elk
    df.loc[df.FIPS == '42049', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # erie
    df.loc[df.FIPS == '42051', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # fayette
    df.loc[df.FIPS == '42053', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # forest
    df.loc[df.FIPS == '42055', c[-4:]] = dt.date(2020, 6, 19).toordinal()  # franklin
    df.loc[df.FIPS == '42057', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # fulton
    df.loc[df.FIPS == '42059', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # greene
    df.loc[df.FIPS == '42061', c[-4:]] = dt.date(2020, 6, 19).toordinal()  # huntingdon
    df.loc[df.FIPS == '42063', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # indiana
    df.loc[df.FIPS == '42065', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # jefferson
    df.loc[df.FIPS == '42067', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # juniata
    df.loc[df.FIPS == '42069', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # lackawannaa
    df.loc[df.FIPS == '42071', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # lancaster
    df.loc[df.FIPS == '42073', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # lawrence
    df.loc[df.FIPS == '42075', c[-4:]] = dt.date(2020, 7, 3).toordinal()  # lebanon####
    df.loc[df.FIPS == '42077', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # lehigh
    df.loc[df.FIPS == '42079', c[-4:]] = dt.date(2020, 6, 19).toordinal()  # lezerne
    df.loc[df.FIPS == '42081', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # lycoming
    df.loc[df.FIPS == '42083', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # mckean
    df.loc[df.FIPS == '42085', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # mercer
    df.loc[df.FIPS == '42087', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # mifflin
    df.loc[df.FIPS == '42089', c[-4:]] = dt.date(2020, 6, 19).toordinal()  # monroe
    df.loc[df.FIPS == '42091', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # montgomery
    df.loc[df.FIPS == '42093', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # montour
    df.loc[df.FIPS == '42095', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # northampton
    df.loc[df.FIPS == '42097', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # northumberland
    df.loc[df.FIPS == '42099', c[-4:]] = dt.date(2020, 6, 19).toordinal()  # perry
    df.loc[df.FIPS == '42101', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # philadelphia
    df.loc[df.FIPS == '42103', c[-4:]] = dt.date(2020, 6, 19).toordinal()  # pike
    df.loc[df.FIPS == '42105', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # potter
    df.loc[df.FIPS == '42107', c[-4:]] = dt.date(2020, 6, 19).toordinal()  # schuylkill
    df.loc[df.FIPS == '42109', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # snyder
    df.loc[df.FIPS == '42111', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # somerset
    df.loc[df.FIPS == '42113', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # sullivan
    df.loc[df.FIPS == '42115', c[-4:]] = dt.date(2020, 6, 26).toordinal()  # susquehanna
    df.loc[df.FIPS == '42117', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # tioga
    df.loc[df.FIPS == '42119', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # union
    df.loc[df.FIPS == '42121', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # venango
    df.loc[df.FIPS == '42123', c[-4:]] = dt.date(2020, 5, 29).toordinal()  # warren
    df.loc[df.FIPS == '42125', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # washington
    df.loc[df.FIPS == '42127', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # wayne  
    df.loc[df.FIPS == '42129', c[-4:]] = dt.date(2020, 6, 5).toordinal()  # westmoreland
    df.loc[df.FIPS == '42131', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # wyoming
    df.loc[df.FIPS == '42133', c[-4:]] = dt.date(2020, 6, 12).toordinal()  # york

    # Puerto Rico PR
    # Probably some info here src: https://www.ddec.pr.gov/covid19_informaciongeneral/ (in spanish)

    # Rhode Island RI
    # South Carolina SC
    #   restaurant dine in : May 11th: src: https://governor.sc.gov/news/2020-05/gov-henry-mcmaster-restaurants-are-able-open-limited-dine-services-monday-may-11
    #   entertainment/gym: May 18th src: https://governor.sc.gov/news/2020-05/gov-henry-mcmaster-announces-additional-businesses-gyms-pools-are-able-open-monday-may
    df.loc[df.STATE == 'SC', c[-2]] = dt.date(2020, 5, 11).toordinal()
    df.loc[df.STATE == 'SC', c[-1]] = dt.date(2020, 5, 18).toordinal()
    # Tennessee TN
    # TODO: County response
    # Texas TX
    # Utah UT
    #   restaurant dine in & entertainment/gym :May 1st src: https://coronavirus-download.utah.gov/Governor/Utah_Leads_Together_3.0_May2020_v20.pdf
    df.loc[df.STATE == 'UT', c[-2]] = dt.date(2020, 5, 1).toordinal()
    df.loc[df.STATE == 'UT', c[-1]] = dt.date(2020, 5, 1).toordinal()
    # Vermont VT 
    # Virginia VA
    # soon: consider phase three starting JUly 1st: src: https://www.governor.virginia.gov/media/governorvirginiagov/governor-of-virginia/pdf/Forward-Virginia-Phase-Three-Guidelines.pdf
    # Washington WA
    # TODO: County response src: https://www.governor.wa.gov/sites/default/files/SafeStartPhasedReopening.pdf
    # West Virginia WV
    #   restaurant Dine in May 4th: src: https://governor.wv.gov/Pages/The-Comeback.aspx
    #   gatherin > 50 June 5th    : src: https://governor.wv.gov/Pages/The-Comeback.aspx
    #   entertainment/gym May 21st: src: https://governor.wv.gov/Pages/The-Comeback.aspx
    #   stay at home order May 3rd: https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html
    df.loc[df.STATE == 'WV', c[-5]] = dt.date(2020, 5, 3).toordinal()
    df.loc[df.STATE == 'WV', c[-4]] = dt.date(2020, 6, 5).toordinal()
    df.loc[df.STATE == 'WV', c[-2]] = dt.date(2020, 5, 4).toordinal()
    df.loc[df.STATE == 'WV', c[-1]] = dt.date(2020, 5, 21).toordinal()
    # Wisconsin WI
    # TODO: county response: src: https://www.wisbank.com/articles/2020/05/wisconsin-county-list-of-safer-at-home-orders/
    #   stay at home May 13th src: https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html
    df.loc[df.STATE == 'WI', c[-5]] = dt.date(2020, 5, 13).toordinal()
    
    
    # Wyoming WY 
    # County Responses: https://www.wyo-wcca.org/index.php/covid-19-resources/emergency-declarations-and-public-building-access/
    # Restaurant order in place from July 1st on src: https://drive.google.com/file/d/1yP1IHC60t9pHQMeenAEzyAuVZJSNAvH2/view
    df.loc[df.STATE == 'WY', c[-2]] = dt.date(2020, 7, 1).toordinal()

    print(df[df['STATE'] == 'WY'].iloc[0])

    # print(df.loc['10000',:])

    print('--------------AFTER--------------')
    # print(df)

    return df


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--data-dir', '-d', default='data', type=str, help='Data directory')

  args = parser.parse_args()

  data_parser = IHMEDataParser(args.data_dir)
