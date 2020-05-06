import pandas as pd

def get_cluster(filename, cluster_num):
    df = pd.read_csv(filename)
    fips = df.loc[df['cluster'] == cluster_num, 'FIPS']
    return fips.values


if __name__ == '__main__':
    filename = 'data/us_data/clustering.csv'
    cluster_num = 5
    fips = get_cluster(filename, cluster_num)
    print(fips)
