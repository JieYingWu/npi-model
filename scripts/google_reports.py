import os
import csv
import wget
import pandas as pd
import numpy as np 

from os.path import join, exists


def download_report(save_path='data/us_data/'):
    print('Downloading latest mobility report..')

    url = 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'
    wget.download(url, join(save_path, 'Global_Mobility_Report.csv'))
    print('Download successful.')

def parse_report(save_path='test/'):
    if not exists(save_path):
        os.makedirs(save_path)
    path = 'data/us_data/Global_Mobility_Report.csv'
    df = pd.read_csv(path)
    df = df[df['country_region_code']=='US']
    dates = list(df['date'].unique())
    print(dates)
    header = df.columns.values
    categories = header[5:]



    fips_dict = {}
    with open('data/us_data/FIPS_lookup.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)

        for row in reader:
            fips_dict['_'.join(row[1:])] = row[0]


    list_of_categories = []
    for i in range(len(categories)):
        list_of_categories.append([])
    
    with open(path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        header = next(reader)
        
        index = -1

        current_county = ''
        for row in reader:
            if row[0] != 'US':
                continue
            if row[2] == '' or row[3] == '':
                continue

            
            if row[3] != current_county:
                current_county = row[3]
                
                for l in list_of_categories:
                    l.append([])
                index += 1
                for j, l in enumerate(list_of_categories, 5):
                    key_row = '_'.join(row[2:4])
                    if key_row not in fips_dict.keys():
                        for key in fips_dict.keys():
                            if key_row == key[:len(key_row)]:
                                key_row = key

                    l[index].append(fips_dict[key_row])
                    l[index].extend(row[2:4])
                    l[index].append(row[j])
            else:
                for j, l in enumerate(list_of_categories, 5):
                    l[index].append(row[j])

    new_paths = [join(save_path,f+'.csv') for f in categories]
    print(new_paths)

    for j, new_path in enumerate(new_paths):
        with open(new_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['FIPS', 'State', 'County'] + dates)
            writer.writerows(list_of_categories[j])
            




if __name__ == '__main__':
    parse_report(save_path='data/us_data/Google_traffic')