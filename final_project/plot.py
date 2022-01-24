from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2


train_df = pd.read_csv("train.csv", encoding="utf-8")

pandas_profiling.ProfileReport(train_df)

y_train = train_df['LEVEL']
x_train = train_df.drop(['LEVEL'], axis=1)
list_columns = train_df.columns.tolist()

plt.style.use("ggplot")
plt.rc('font', family='SimHei', size=13)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
cat_list = list_columns
for n, i in enumerate(cat_list):  
    Cabin_cat_num = train_df[i].value_counts().index.shape[0]
    print('{0}. {1}特征的类型数量是: {2}'.format(n+1, i, Cabin_cat_num))

f, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
sns.countplot(x='Season', hue='LEVEL', data=train_df, ax=ax1)
sns.countplot(x='海岸段', hue='LEVEL', data=train_df, ax=ax2)
sns.countplot(x='Region', hue='LEVEL', data=train_df, ax=ax3)
ax1.set_title('Season feature analysis')
ax2.set_title('Coastal feature analysis')
ax3.set_title('Region feature analysis')

f, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
sns.countplot(x='Seat', hue='LEVEL', data=train_df, ax=ax1)
sns.countplot(x='Shore shape', hue='LEVEL', data=train_df, ax=ax2)
sns.countplot(x='Substrate type', hue='LEVEL', data=train_df, ax=ax3)
ax1.set_title('Seat feature')
ax2.set_title('Coastal feature ')
ax3.set_title('Substrate_type feature')

f, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
sns.countplot(x='1暴露岩岸', hue='LEVEL', data=train_df, ax=ax1)
sns.countplot(x='2暴露人造結構物', hue='LEVEL', data=train_df, ax=ax2)
sns.countplot(x='3暴露岩盤', hue='LEVEL', data=train_df, ax=ax3)
ax1.set_title('1.Rock Coast')
ax2.set_title('2.Exposed artificial structure')
ax3.set_title('3.Exposed bedrock')

f, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
sns.countplot(x='4沙灘', hue='LEVEL', data=train_df, ax=ax1)
sns.countplot(x='5砂礫混合灘', hue='LEVEL', data=train_df, ax=ax2)
sns.countplot(x='6礫石灘', hue='LEVEL', data=train_df, ax=ax3)
ax1.set_title('4.Beach feature')
ax2.set_title('5.Gravel mixed beach feature')
ax3.set_title('6.Gravel beach feature')

f, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
sns.countplot(x='7開闊潮間帶', hue='LEVEL', data=train_df, ax=ax1)
sns.countplot(x='8遮蔽岩岸', hue='LEVEL', data=train_df, ax=ax2)
sns.countplot(x='9遮蔽潮間帶', hue='LEVEL', data=train_df, ax=ax3)
ax1.set_title('7.Broad intertidal zone')
ax2.set_title('8.Shaded rock coast')
ax3.set_title('9.Shaded intertidal zone')

f, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
sns.countplot(x='10遮蔽濕地', hue='LEVEL', data=train_df, ax=ax1)
sns.countplot(x='Plastic bottle container', hue='LEVEL', data=train_df, ax=ax2)
sns.countplot(x='Disposable cup / straw / tableware', hue='LEVEL', data=train_df, ax=ax3)
ax1.set_title('10.Shaded wetlands')
ax2.set_title('Plastic_bottle_container feature analysis')
ax3.set_title('Disposable_cup_/_straw_/_tableware feature analysis')

f, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
sns.countplot(x='Plastic bag', hue='LEVEL', data=train_df, ax=ax1)
sns.countplot(x='Foam material', hue='LEVEL', data=train_df, ax=ax2)
sns.countplot(x='Float', hue='LEVEL', data=train_df, ax=ax3)
ax1.set_title('Plastic_bag feature analysis')
ax2.set_title('Foam_material feature analysis')
ax3.set_title('Float feature analysis')

f, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
sns.countplot(x='Fishing nets and ropes', hue='LEVEL', data=train_df, ax=ax1)
sns.countplot(x='Fishing equipment', hue='LEVEL', data=train_df, ax=ax2)
sns.countplot(x='Cigarette and lighter', hue='LEVEL', data=train_df, ax=ax3)
ax1.set_title('Fishing_nets_and_ropes feature analysis')
ax2.set_title('Fishing_equipment feature analysis')
ax3.set_title('Cigarette_and_lighter feature analysis')

f, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
sns.countplot(x='Glass jar', hue='LEVEL', data=train_df, ax=ax1)
sns.countplot(x='Metal', hue='LEVEL', data=train_df, ax=ax2)
sns.countplot(x='Paper', hue='LEVEL', data=train_df, ax=ax3)
ax1.set_title('Glass_jar feature analysis')
ax2.set_title('Metal feature analysis')
ax3.set_title('Paper feature analysis')

f, [ax1,ax2] = plt.subplots(1,2,figsize=(20,5))
sns.countplot(x='Others', hue='LEVEL', data=train_df, ax=ax1)
sns.countplot(x='LEVEL', hue='LEVEL', data=train_df, ax=ax2)
ax1.set_title('Others feature analysis')
ax2.set_title('LEVEL feature analysis')

plt.show()
