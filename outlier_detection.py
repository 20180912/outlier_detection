#importing relevant packages
import pandas as pd
import matplotlib.pyplot as plt
#importing the data set
filename = 'train.gz'
 
chunksize = 100000
chunks = []
for chunk in pd.read_csv(filename, compression='gzip', header=0, sep=',', usecols= ['hour', 'click'], index_col='hour', chunksize=chunksize):
    #broadcasting 1 to count impressions. If totals aren't needed, the CTR can be calulated by taking the mean of clicks
    chunk['impression'] = 1
    #converting the index column to datetime and resampling
    time_format = '%y%m%d%H'
    chunk.index = pd.to_datetime(chunk.index, format=time_format)
    chunk = chunk.resample('H').sum()
    chunks.append(chunk)
 
df = pd.concat(chunks)
df = df.groupby(df.index).sum()
 
"""
#The following code works as well and is simpler but requires more RAM
df = pd.read_csv(filename, compression='gzip', header=0, sep=',', usecols= ['hour', 'click'], index_col='hour')
df['impression'] = 1
time_format = '%y%m%d%H'
df.index = pd.to_datetime(df.index, format=time_format)
df=df.resample('H').sum()
"""
 
df['CTR'] = df['click']/df['impression']
#calculating moving average with beginning fixed, alternatively use df['CTR'].rolling(window=...).mean() for a constant time frame
df['CTR_moving_avg'] = df['CTR'].expanding().mean()
df['CTR_moving_std'] = df['CTR'].expanding().std()
#boolean indexing to capture only outliers in new column
mask = abs(df['CTR']-df['CTR_moving_avg'])>1.5*df['CTR_moving_std']
df['outlier'] = df.loc[mask,'CTR']
 
#task No.1
plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w')
plt.subplot(2,1,1)
plt.title('CTR')
plt.plot(df['CTR'], color='k')
plt.xticks(rotation=30)
plt.ylabel('CTR')
plt.ylim(bottom=0)
plt.style.use('ggplot')
#task No.2
plt.subplot(2,1,2)
plt.plot(df['CTR'], color='k', linestyle = 'none', marker = '.')
plt.plot(df['CTR_moving_avg'], color='b', linestyle='dashed')
plt.plot(df['outlier'], color='r', linestyle = 'none', marker = '.')
plt.fill_between(df.index,df['CTR_moving_avg']+1.5*df['CTR_moving_std'], df['CTR_moving_avg']-1.5*df['CTR_moving_std'], color='b', alpha=0.3)
plt.tight_layout(pad=4, h_pad=4)
#using the default x-axis. It should be possible to create different lables using df.index.strftime('...')
plt.xticks(rotation=30)
plt.title('Outliers')
plt.legend(('CTR','$\overline{CTR}$', 'Outlier', '$1.5\cdot\sigma$'), loc='best')
plt.xlabel('day')
plt.ylabel('CTR')
plt.ylim(bottom=0)
plt.style.use('ggplot')

#saving and plotting the charts 
plt.savefig('output.png')
plt.show()