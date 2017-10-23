import pandas as pd
from Utility.importer import importer
from sklearn.cluster import KMeans

# run the importer on the data folder
#df = importer('data') 
# send it to a TSV for Tableau visualisation
#df.to_csv('output-rawcounters.csv')

# Unzip this file and load from here
df = pd.DataFrame.from_csv('output-rawcounters.csv')

# This groups by userid and level_int, using mean as the aggregate.
# Pandas defaults to making these groupings the indexes of the new table, so we reset that
udf = df.groupby(['userid', 'level_int']).aggregate('mean')
udf = udf.reset_index([0,1])

# This time we want just the userids, and we want to drop all the other data. We reset the index again as before
uids = df.groupby(['userid']).aggregate('mean')
uids = uids.drop(['duration_float','level_int','defeated_enemies_int','collected_gold_int','remaining_life_int'],1)
uids = uids.reset_index(0)

# initialise a new DF with all the userids
combined = uids
# for every level
for i in range(1,51):
    # select the subset of the data that's for that data
    subset = udf[(udf.level_int == i)]
    # we know which level this is so we can drop this in the subset
    subset = subset.drop('level_int',1)
    # but we do want to annotate our column names with the level number
    subset = subset.add_prefix(str(i)+"_")
    # except not the userid column, so set that one back
    subset = subset.rename(columns={str(i)+"_userid":"userid"})
    # this works like an SQL join - combined is joined to subset on userid.
    # we do an outer join to retain all the userids and put nulls (technically NaNs) is there's no data 
    combined = combined.merge(subset,on='userid',how='outer')

# Remove userIDs    
clusterData = combined.drop('userid',1)
# Replace NaN with column means
clusterData = clusterData.fillna(clusterData.mean())

# We turn the DF into a numPy matrix so KMeans can work with it
mat = clusterData.as_matrix()

# Now we can cluster
km = KMeans(n_clusters=20,init='random')
labels = km.fit_predict(mat)

# TODO work out how to action the clustering algorithm
clustering = pd.Series(labels)
clusterData['cluster'] = clustering.values
clusterData.to_csv('output-clustered.csv')
# Send to CSV one last time to visualise again

