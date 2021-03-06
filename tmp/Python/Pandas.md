---
title: "Pandas"
description: "Pandas"
url: pandas2
---

# Pandas

# Basic

```python
import pandas as pd

##read file

TB = pd.read_csv("sentence-G",sep='\t', names=None, header=None)
TB.columns = ['频率', '', 'V2', '词义', 'V3']

#DataFrame
pd.DataFrame(index=range(40),columns=['a', 'b'])

## rsult output
TB.to_csv("table",sep='\t')

data
data[0:2]       #取前两行数据

len(data )              #求出一共多少行
data.columns.size      #求出一共多少列
data.columns        #列索引名称
data.index       #行索引名称

data.ix[1]                #取第2行数据
data.iloc[1]             #取第2行数据
data.loc['A']      #取第行索引为”A“的一行数据，
data['x']      #取列索引为x的一列数据

data.loc[:,['x','z'] ]          #表示选取所有的行以及columns为a,b的列；
data.loc[['A','B'],['x','z']]     #表示选取'A'和'B'这两行以及columns为x,z的列的并集；
data.iloc[1:3,1:3]              #数据切片操作，切连续的数据块
data.iloc[[0,2],[1,2]]              #即可以自由选取行位置，和列位置对应的数据，切零散的数据块

data[data>2]       #表示选取数据集中大于0的数据
data[data.x>5]       #表示选取数据集中x这一列大于5的所有的行
a1=data.copy()
a1[a1['y'].isin(['6','10'])]    #表显示满足条件：列y中的值包含'6','8'的所有行。

data.mean()           #默认对每一列的数据求平均值；若加上参数a.mean(1)则对每一行求平均值；
data['x'].value_counts()    #统计某一列x中各个值出现的次数：
data.describe() #对每一列数据进行统计，包括计数，均值，std，各个分位数等。
data.to_excel(r'E:\pypractice\Yun\doc\2.xls',sheet_name='Sheet1')  #数据输出至Exceldata[0:2]       #取前两行数据

## from Dictionary to DataFrame
TB = pd.Series(BB)

## DataFrame sort
TB = TB.sort_values(ascending=False)

## DataFrame merge
result = pd.concat([ Word, Sen], axis=1, sort=False)

## NaN drap
data.dropna(thresh=3) # at least 3 data we have
```

# Skills
reference: [数据分析1480](https://mp.weixin.qq.com/s/Dm-pP6o4_qRQ49mnzFUcIQ)

##  NA count
```python
df=pd.read_csv('titanic_train.csv')
def missing_cal(df):
    """
    df :数据集  
    return：每个变量的缺失率
    """
    missing_series = df.isnull().sum()/df.shape[0]
    missing_df = pd.DataFrame(missing_series).reset_index()
    missing_df = missing_df.rename(columns={'index':'col',
                                            0:'missing_pct'})
    missing_df = missing_df.sort_values('missing_pct',ascending=False).reset_index(drop=True)
    return missing_df
missing_cal(df)
```

## idmax
```python
df = pd.DataFrame({'Sp':['a','b','c','d','e','f'], 'Mt':['s1', 's1', 's2','s2','s2','s3'], 'Value':[1,2,3,4,5,6], 'Count':[3,2,5,10,10,6]})
df

df.iloc[df.groupby(['Mt']).apply(lambda x: x['Count'].idxmax())]

df["rank"] = df.groupby("ID")["score"].rank(method="min", ascending=False).astype(np.int64)
df[df["rank"] == 1][["ID", "class"]]
```

## Raw merge

```python
df = pd.DataFrame({'id_part':['a','b','c','d'], 'pred':[0.1,0.2,0.3,0.4], 'pred_class':['women','man','cat','dog'], 'v_id':['d1','d2','d3','d1']})

df.groupby(['v_id']).agg({'pred_class': [', '.join],'pred': lambda x: list(x),
'id_part': 'first'}).reset_index()
```

## Deleting rows by string-match

```python
df = pd.DataFrame({'a':[1,2,3,4], 'b':['s1', 'exp_s2', 's3','exps4'], 'c':[5,6,7,8], 'd':[3,2,5,10]})
df[df['b'].str.contains('exp')]
```

## Sort
```python
df = pd.DataFrame([['A',1],['A',3],['A',2],['B',5],['B',9]], columns = ['name','score'])

df.sort_values(['name','score'], ascending = [True,False])
df.groupby('name').apply(lambda x: x.sort_values('score', ascending=False)).reset_index(drop=True)
```

## Select columns by features

```python
drinks = pd.read_csv('data/drinks.csv')
# 选择所有数值型的列
drinks.select_dtypes(include=['number']).head()
# 选择所有字符型的列
drinks.select_dtypes(include=['object']).head()
drinks.select_dtypes(include=['number','object','category','datetime']).head()
# 用 exclude 关键字排除指定的数据类型
drinks.select_dtypes(exclude=['number']).head()
```

## str to integer

```python
df = pd.DataFrame({'列1':['1.1','2.2','3.3'],
                  '列2':['4.4','5.5','6.6'],
                  '列3':['7.7','8.8','-']})
df
df.astype({'列1':'float','列2':'float'}).dtypes


df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
```

## Reduce the RAM-consume

```python
cols = ['beer_servings','continent']
small_drinks = pd.read_csv('data/drinks.csv', usecols=cols)
```

```python
dtypes ={'continent':'category'}
smaller_drinks = pd.read_csv('data/drinks.csv',usecols=cols, dtype=dtypes)
```

## 根据最大的类别筛选 DataFrame
```python
movies = pd.read_csv('data/imdb_1000.csv')
counts = movies.genre.value_counts()
movies[movies.genre.isin(counts.nlargest(3).index)].head()
```

## split string to columns
```python
df = pd.DataFrame({'姓名':['张 三','李 四','王 五'],
                   '所在地':['北京-东城区','上海-黄浦区','广州-白云区']})
df
df.姓名.str.split(' ', expand=True)
```

## 把 Series 里的列表转换为 DataFrame
```python
df = pd.DataFrame({'列1':['a','b','c'],'列2':[[10,20], [20,30], [30,40]]})
df

df_new = df.列2.apply(pd.Series)
pd.concat([df,df_new], axis='columns')
```

## 用多个函数聚合
```python
orders = pd.read_csv('data/chipotle.tsv', sep='\t')
orders.groupby('order_id').item_price.agg(['sum','count']).head()
```

## 分组聚合

```python
import pandas as pd
df = pd.DataFrame({'key1':['a', 'a', 'b', 'b', 'a'],
    'key2':['one', 'two', 'one', 'two', 'one'],
    'data1':np.random.randn(5),
     'data2':np.random.randn(5)})
df

for name, group in df.groupby('key1'):
    print(name)
    print(group)

dict(list(df.groupby('key1')))
```

# 通过字典或Series进行分组

```python
people = pd.DataFrame(np.random.randn(5, 5),
     columns=['a', 'b', 'c', 'd', 'e'],
     index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
mapping = {'a':'red', 'b':'red', 'c':'blue',
     'd':'blue', 'e':'red', 'f':'orange'}
by_column = people.groupby(mapping, axis=1)
by_column.sum()
```


---
github: [https://github.com/Karobben](https://github.com/Karobben)
blog: [Karobben.github.io](http://Karobben.github.io)
R 语言画图索引: [https://karobben.github.io/R/R-index.html](https://karobben.github.io/R/R-index.html)
