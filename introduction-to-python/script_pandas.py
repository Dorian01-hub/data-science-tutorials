
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data/train.csv')

print(df.head())

print(df.isnull().sum())



print(df.groupby('Survived')['PassengerId'].agg(['count']))






df.groupby('Survived')['PassengerId'].agg(
    ['count']
).reset_index().plot(x='Survived', y='count', kind = 'bar', figsize = (10, 10))

plt.show()




print(df.groupby(
    ['Survived', 'Sex']
)['PassengerId'].agg(['count']))





print(df.groupby(
    ['Survived', 'Sex']
)['PassengerId'].agg(['count']).unstack())





df.groupby(
    ['Survived', 'Sex']
)['PassengerId'].count().unstack().plot(kind ='bar', figsize = (10, 10))


plt.show()




print(df.groupby(['Survived', 'Pclass'])['PassengerId'].agg(
    ['count']
))




print(
df.groupby(['Survived', 'Pclass'])['PassengerId'].count().unstack())




df.groupby(
    ['Survived', 'Pclass']
)['PassengerId'].count().unstack().plot(kind ='bar', figsize = (10, 10))


plt.show()






df['generation'] = pd.cut(df['Age'], 8)




pd.cut(df['Age'], 8)



print(
df.head())





df.groupby(
    ['Survived', 'generation']
)['PassengerId'].count().unstack().plot(kind ='bar', figsize = (10, 10))

plt.show()



df['fare_category'] = pd.cut(df['Fare'], 12)




pd.cut(df['Fare'], 10)






df.groupby(
    ['Survived', 'fare_category']
)['PassengerId'].count().unstack().plot(kind ='bar', figsize = (10, 10))
plt.show()




print(
df[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']].corr())





df[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']].corr().style.background_gradient(cmap='coolwarm')
plt.show()



### Si on met qu'un plt.show() à la fin, on affiche toutes les figures d'un coup et pas une à une comme ici




# ##### Cabins on the port side have an even number and cabins on the starboard side have an odd number. 
# 
# #### For example, cabin B57 is located on the starboard side.
# 
# #### Which side of the boat is better to be on? 
# 
# 
# #### The deck number of the boat is indicated on the ticket. Cabin B57 is located on deck B. Which deck is best to be on?
# 
# #### Where is the best place to be on the boat in general?
# 
# #### Is there a link between the number of parents/family on the boat and chances of survival?

# %% [markdown]
# #### What is the typical profile of the person who will survive the shipwreck?
# 
# #### What is the typical profile of the person who will not survive the shipwreck?


