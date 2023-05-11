import csv
import pyspark
import pyproj
import pandas as pd
import sys
from pyspark.sql import SparkSession
sc = pyspark.SparkContext.getOrCreate()
spark = SparkSession(sc)

def extract(partId, rows):
    if partId==0:
        next(rows)
    reader = csv.reader(rows)
    for row in reader:
      if not row[19] == '{}':
        if row[0] in safegraph_placekey:
          (poi_cbg, visitor_cbg, date_start, date_end) = (row[18],row[19][1:-1],row[12][0:7],row[13][0:7]) 
          yield (poi_cbg, visitor_cbg, date_start, date_end)

def merge(x, y):
    z = x.copy()   
    z.update(y)    
    return z      

if __name__=="__main__":
  sc = pyspark.SparkContext.getOrCreate()
  spark = SparkSession(sc)

  WEEKPATTERN_DF= '/shared/CUSP-GX-6002/data/weekly-patterns-nyc-2019-2020/*'
  SUPERMARKET_DF = pd.read_csv('nyc_supermarkets.csv',encoding="utf-8")
  CENTROIDS_DF = pd.read_csv('nyc_cbg_centroids.csv',encoding="utf-8")

  safegraph_placekey = spark.createDataFrame(SUPERMARKET_DF)\
                            .rdd\
                            .map(lambda x: {x[-2]})\
                            .reduce(lambda x,y: x|y)

  proj = pyproj.Proj(init='EPSG:2263', preserve_units=False)
  centroid = spark.createDataFrame(CENTROIDS_DF)\
                  .rdd\
                  .map(lambda x: (x[0],proj(x[2],x[1])))\
                  .map(lambda x: (str(x[0]),(x[1][0]/1609.344,x[1][1]/1609.344)))

  A = sc.textFile(WEEKPATTERN_DF, use_unicode=True)

  date_range = ['2019-03', '2019-10', '2020-03', '2020-10']
  B = A.mapPartitionsWithIndex(extract)\
       .filter(lambda x: x[2] in date_range or x[3] in date_range)\
       .map(lambda x:(x[0],x[1].split(','),x[2]) if x[2] in date_range else(x[0],x[1].split(','),x[3]))

  C = B.flatMap(lambda x: [(x[0],x[2],b.split(':')[0][1:-1],int(b.split(':')[1])) for b in x[1]])\
       .filter(lambda x: x[0].startswith('36') and x[2].startswith('36'))\
       .map(lambda x: (x[0],(x[1],x[2],x[3])))\
       .leftOuterJoin(centroid)\
       .map(lambda x: (x[1][0][1],(x[0],x[1][1],x[1][0][0],x[1][0][2])))\
       .leftOuterJoin(centroid)\
       .map(lambda x: (x[0],x[1][0][2],x[1][0][3],x[1][0][1],x[1][1]))\
       .filter(lambda x: (not x[3] == None) and (not x[4] == None))\
       .map(lambda x: ((x[0],x[1]),(x[2],x[2]*((x[3][0]-x[4][0])**2+(x[3][1]-x[4][1])**2)**0.5)))\
       .reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1]))\
       .map(lambda x: (x[0][0],x[0][1],round(x[1][1]/x[1][0],2)))

  dict = {'2019-03':'', '2019-10':'', '2020-03':'', '2020-10':''}
  D = C.map(lambda x:(x[0],{x[1]:x[2]}))\
       .reduceByKey(lambda x,y: merge(x,y))\
       .map(lambda x: (x[0],merge(dict,x[1])))\
       .map(lambda x: (int(x[0]),(x[1]['2019-03'], x[1]['2019-10'],x[1]['2020-03'],x[1]['2020-10'])))\
       .sortByKey()\
       .map(lambda x: (x[0],x[1][0],x[1][1],x[1][2],x[1][3]))

  head=sc.parallelize([('cbg_fips','2019-03','2019-10','2020-03','2019-10')])
  outputFinal = head.union(D).map(lambda x: ','.join(str(d) for d in x))
  outputFinal.saveAsTextFile(sys.argv[1])
