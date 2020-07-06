import os 
import wget
import argparse 
import datetime
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
        int_df = pd.read_csv(interventions_path, engine='python', dtype={'FIPS':str})
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

        #select all states of the US in IHME
        ihme_df = ihme_df[ihme_df['location_name'].isin(states)]

        #relevant interventions
        interventions = [#'travel_limit_end_date',
                        'stay_home_end_date',
                        #'educational_fac_end_date',
                        'any_gathering_restrict_end_date',
                        #'any_business_end_date',
                        'all_non-ess_business_end_date'
                        ]
        ihme_df = ihme_df[['location_name']+interventions]

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
        pr = ihme_df[ihme_df['location_name']=='Puerto Rico'].copy()
        ihme_df = ihme_df.drop(253) # index of pr
        ihme_df = ihme_df.append(pr, ignore_index=True)


        # fips state list
        fips_state_list = list(np.linspace(1000,56000, 56, dtype=np.int))
        # fips_state_id_list = 
        fips_state_list += [72000] # puerto rico is special, why can't they vote btw

        
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
        interventions_to_columns_dict={x[0]: c[0],
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

            
            date_current_list = ihme_df[ihme_df['FIPS']==id_][interventions]
            date_current_list = date_current_list.values.tolist()

            for j in range(len(interventions)):
                date_current = date_current_list[0][j]
                if date_current != [] and type(date_current) != np.float:
                    date_current = date_current

                    # parse date
                    date_current_ordinal = datetime.datetime.strptime(date_current, '%Y-%m-%d').toordinal()
                    
                    int_df.at[i, interventions_to_columns_dict[interventions[j]]] = date_current_ordinal

                else:
                    int_df.at[i,interventions_to_columns_dict[interventions[j]]] = np.nan



        int_df = self.dirty_helper(int_df, columns_list)
        int_df.to_csv(join(self.data_dir, 'us_data', 'interventions_updated.csv'),index=True)






    def dirty_helper(self, df, columns_list):
        """ Hardcoded county rollback dates """
        print(df)

        df.set_index('FIPS', inplace=True)
        print('--------------BEFORE--------------')
        #print(df)
        columns_list = ['stay at home rollback',
                       '>50 gatherings rollback',
                       '>500 gatherings rollback',
                       'restaurant dine-in rollback',
                       'entertainment/gym rollback']
        c = columns_list
        # Alabama
        # Alaska
        # Arizona
        # Arkansas
        # California
        # Colorado
        # Conneticut
        # Delaware
        #    restaurant and entertainment open from June 15th on src: https://coronavirus.delaware.gov/reopening/phase2/
        df.loc[df.STATE=='DE', c[-1]] = datetime.date(2020,6,15).toordinal()
        df.loc[df.STATE=='DE', c[-2]] = datetime.date(2020,6,15).toordinal()

        # DC 
        #   stay at home order June 22nd src: https://coronavirus.dc.gov/phasetwo
        df.loc[df.STATE=='DC', c[-5]] = datetime.date(2020,6,22).toordinal()
        
        # FL - checked for counties: phase 1 (restaurant and gym) all took effect at same time
        #   stay at home order May 18 src: https://floridahealthcovid19.gov/plan-for-floridas-recovery/
        #   reastaurants May 18th 
        #   entertainment May 18th
        df.loc[df.STATE=='FL', c[-5]] = datetime.date(2020,6,18).toordinal()
        df.loc[df.STATE=='FL', c[-2]] = datetime.date(2020,6,18).toordinal()
        df.loc[df.STATE=='FL', c[-1]] = datetime.date(2020,6,18).toordinal()


        # Georgia 
        # Hawaii
        # Idaho
        #   gatherings <50 May 30th src: https://rebound.idaho.gov/stages-of-reopening/
        #   gatherings <500 June 13th src: https://rebound.idaho.gov/stages-of-reopening/
        df.loc[df.STATE=='ID', c[-4]] = datetime.date(2020,5,30).toordinal()
        df.loc[df.STATE=='ID', c[-3]] = datetime.date(2020,6,13).toordinal()

        # Illinois
        # stay at home order May 30th src:https://www.pantagraph.com/news/state-and-regional/illinois-stay-at-home-order-ends-and-restrictions-lifted-on-churches-as-the-state-advances/article_71393207-40a5-58cf-a658-c580da3d437d.html
        df.loc[df.STATE=='IL', c[-5]] = datetime.date(2020,5,30).toordinal()
        
        # Indiana 
        # Iowa
        # https://wcfcourier.com/news/local/govt-and-politics/update-watch-now-iowa-to-reopen-restaurants-friday/article_7636be19-9dec-5cb9-8344-29c6aafd0196.html
        df.loc[df.FIPS=='19153', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19005', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19011', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19013', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19017', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19049', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19057', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19061', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19065', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19087', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19095', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19099', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19103', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19113', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19115', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19127', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19139', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19157', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19163', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19171', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19183', c[-2:]] = datetime.date(2020,5,15).toordinal()
        df.loc[df.FIPS=='19193', c[-2:]] = datetime.date(2020,5,15).toordinal()
                
        
        # Kansas 
        # Kentucky
        # Lousiana
        # indoor limit <250 people not considered: src: https://gov.louisiana.gov/index.cfm/newsroom/detail/2573
        # Maine
        # Maryland https://conduitstreet.mdcounties.org/2020/05/15/marylands-reopening-status-by-county/
        # Anne Arundel County https://www.aacounty.org/coronavirus/road-to-recovery/
        df.loc[df.FIPS=='24003', c[-2:]] = datetime.date(2020,5,29).toordinal() 
        df.loc[df.FIPS=='24003', c[-1:]] = datetime.date(2020,6,4).toordinal() 
        
        # Baltimore https://baltimore.cbslocal.com/reopening-maryland-whats-open-whats-closed-county-by-county/
        df.loc[df.FIPS=='24510', c[-2:]] = datetime.date(2020,5,29).toordinal() 
        df.loc[df.FIPS=='24510', c[-1:]] = datetime.date(2020,6,19).toordinal() 

        # Baltimore County https://www.baltimorecountymd.gov/News/BaltimoreCountyNow/baltimore-county-to-fully-enter-stage-one-reopening
        df.loc[df.FIPS=='24005', c[-2:]] = datetime.date(2020,5,29).toordinal()
        df.loc[df.FIPS=='24005', c[-1:]] = datetime.date(2020,6,5).toordinal()

        # Charles County https://www.charlescountymd.gov/services/health-and-human-services/covid-19
        df.loc[df.FIPS=='24005', c[-2:]] = datetime.date(2020,5,29).toordinal()
        df.loc[df.FIPS=='24005', c[-1:]] = datetime.date(2020,6,19).toordinal()

        # Frederick County https://health.frederickcountymd.gov/621/Recovery
        df.loc[df.FIPS=='24021', c[-2:]] = datetime.date(2020,5,29).toordinal()
        df.loc[df.FIPS=='24021', c[-1:]] = datetime.date(2020,6,19).toordinal()

        # Howard County https://www.howardcountymd.gov/News/ArticleID/2007/Coronavirus-Updates-Howard-County-Aligns-with-Governor%E2%80%99s-Phase-2-Reopening-Contact-Tracing-Campaign
        df.loc[df.FIPS=='24027', c[-2:]] = datetime.date(2020,5,29).toordinal()
        df.loc[df.FIPS=='24027', c[-1:]] = datetime.date(2020,6,5).toordinal()

        # Montgomery County https://www.montgomerycountymd.gov/covid19/news/index.html
        df.loc[df.FIPS=='24031', c[-2:]] = datetime.date(2020,6,1).toordinal()
        df.loc[df.FIPS=='24031', c[-1:]] = datetime.date(2020,6,19).toordinal()

        # Prince George County https://www.princegeorgescountymd.gov/Archive.aspx?AMID=142
        df.loc[df.FIPS=='24033', c[-2:]] = datetime.date(2020,6,1).toordinal()
        df.loc[df.FIPS=='24033', c[-1:]] = datetime.date(2020,6,29).toordinal()
        
        # Massachusetts
        #   restaurants: June 22nd src: https://www.mass.gov/info-details/safety-standards-and-checklist-restaurants
        df.loc[df.STATE=='MA', c[-2]] = datetime.date(2020,6,22).toordinal()

        print(df[df['STATE']=='MA'].iloc[0])

        # print(df.loc['10000',:])
        
        


        print('--------------AFTER--------------')
        #print(df)

        return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', '-d', default='data', type=str, help='Data directory')
     
    args = parser.parse_args()

    data_parser = IHMEDataParser(args.data_dir)
