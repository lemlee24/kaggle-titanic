### Kaggle 竞赛 | 泰坦尼克号灾难中的机器学习

> RMS 泰坦尼克号（RMS Titanic）的沉没是历史上最臭名昭著的海难之一。1912 年 4 月 15 日，在它的处女航中，泰坦尼克号与冰山相撞后沉没，2224 名乘客和船员中有 1502 人遇难。这一骇人听闻的悲剧震惊了国际社会，并促成了更严格的船舶安全法规。
>
> 这次海难之所以造成如此巨大的人员伤亡，其中一个原因是船上没有为所有乘客和船员准备足够的救生艇。虽然在这次沉船中生还与否有一定的运气成分，但有些群体比其他人更有可能生还，比如女性、儿童以及上层阶级。
>
> 在本次竞赛中，我们希望你完成分析：什么样的人更有可能在这场灾难中存活。更具体地说，我们希望你运用机器学习工具，预测哪些乘客在这场悲剧中幸存。
>
> 这是一场 Kaggle 的入门级竞赛，非常适合对数据科学和机器学习缺乏经验的人作为起点。

来自竞赛的[主页](http://www.kaggle.com/c/titanic-gettingStarted)。


### 本 Notebook 的目标

展示一个用完整 PyData 工具链，在 Python 中对泰坦尼克号灾难进行分析的简单示例。本 Notebook 面向想要入门该领域的人，或者已经在该领域、想看看一个用 Python 完成分析示例的人。

#### 本 Notebook 将展示以下基础示例：

#### 数据处理
- 导入 Pandas 数据
- 清洗数据
- 使用 Matplotlib 进行可视化探索

#### 数据分析
- 监督式机器学习技术：
  - 逻辑回归模型（Logit Regression）
  - 绘制结果
  - 使用 3 种核函数的支持向量机（SVM）
  - 基础随机森林（Random Forest）
  - 绘制结果

#### 分析评估
- 使用 K 折交叉验证在本地评估结果
- 将 Notebook 的结果输出到 Kaggle


#### 所需库
- [NumPy](http://www.numpy.org/)
- [IPython](http://ipython.org/)
- [Pandas](http://pandas.pydata.org/)
- [SciKit-Learn](http://scikit-learn.org/stable/)
- [SciPy](http://www.scipy.org/)
- [StatsModels](http://statsmodels.sourceforge.net/)
- [Patsy](http://patsy.readthedocs.org/en/latest/)
- [Matplotlib](http://matplotlib.org/)

***要交互式运行这个 Notebook，可以从我的 Github 获取[这里](https://github.com/agconti/kaggle-titanic)。竞赛网站在 [Kaggle.com](http://www.kaggle.com/c/titanic-gettingStarted)。***

```python
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm
from KaggleAux import predict as ka  # see github.com/agconti/kaggleaux for more details
```

### 数据处理

#### 使用 pandas 读取数据：

```python
df = pd.read_csv("data/train.csv")
```

展示数据概览：

```python
df
```

### 来看一下数据

上面是我们数据的概要，它保存在一个 `Pandas` 的 `DataFrame` 中。你可以把 `DataFrame` 看作是 Python 版、功能大幅增强的 Excel 表格工作流。如你所见，这个概要包含了相当多的信息。

首先，它告诉我们有 891 条观测（也就是 891 名乘客）可供分析：

- `Int64Index: 891 entries, 0 to 890`

接着，它展示了 `DataFrame` 中所有的列。每一列都告诉我们每条观测的一些信息，比如乘客的 `name`、`sex` 或 `age`。这些列就是我们数据集的特征（feature）。在本 Notebook 中，可以把“列（column）”和“特征（feature）”这两个词的含义看作是可互换的。

在每个特征后面，会告诉我们该特征包含多少个值。比如大多数特征对每一条观测都有完整的数据，比如这里的 `survived` 特征：

- `survived    891  non-null values`

但也有些特征缺失了一部分信息，比如 `age`：

- `age         714  non-null values`

这些缺失值在数据中表现为 `NaN`。

### 处理缺失值

特征 `ticket` 和 `cabin` 有大量缺失值，因此对我们的分析帮助不大。为了保证数据集的质量，我们会直接删除这两个特征。

为此，我们用下面这行代码从 DataFrame 中完全删除这些特征：

```python
df = df.drop(['ticket','cabin'], axis=1)
```

而下面这行代码则会删除剩余列中仍带有 `NaN` 的那些观测：

```python
df = df.dropna()
```

现在，我们得到了一个干净、整齐的数据集，可以开始分析了。注意：因为 `.dropna()` 会在任意一个特征存在 `NaN` 时就删除整条观测，所以如果我们之前不先删除 `ticket` 和 `cabin` 特征，它会删掉我们大部分的数据。

在本 Notebook 中，对应的代码是：

```python
df = df.drop(['Ticket','Cabin'], axis=1)
# Remove NaN values
df = df.dropna()
```

关于如何使用 pandas 进行数据分析，更详细的介绍可以参考 Wes McKinney 的[这本书](http://shop.oreilly.com/product/0636920023784.do)。此外，一些涵盖基础内容的交互式教程可以在[这里](https://bitbucket.org/hrojas/learn-pandas)找到（免费）。如果你还对 pandas 的强大心存怀疑，可以看看这篇旋风式[简介](http://wesmckinney.com/blog/?p=647)，了解它都能做些什么。

### 图形化审视数据

```python
# specifies the parameters of our graphs
fig = plt.figure(figsize=(18,6), dpi=1600)
alpha = alpha_scatterplot = 0.2
alpha_bar_chart = 0.55

# lets us plot many diffrent shaped graphs together
ax1 = plt.subplot2grid((2,3),(0,0))
# plots a bar graph of those who surived vs those who did not.
df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
ax1.set_xlim(-1, 2)
# puts a title on our graph
plt.title("Distribution of Survival, (1 = Survived)")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age, alpha=alpha_scatterplot)
# sets the y axis lable
plt.ylabel("Age")
# formats the grid line style of our graphs
plt.grid(b=True, which='major', axis='y')
plt.title("Survival by Age,  (1 = Survived)")

ax3 = plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
ax3.set_ylim(-1, len(df.Pclass.value_counts()))
plt.title("Class Distribution")

plt.subplot2grid((2,3),(1,0), colspan=2)
# plots a kernel density estimate of the subset of the 1st class passangers's age
df.Age[df.Pclass == 1].plot(kind='kde')
df.Age[df.Pclass == 2].plot(kind='kde')
df.Age[df.Pclass == 3].plot(kind='kde')
# plots an axis lable
plt.xlabel("Age")
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')

ax5 = plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax5.set_xlim(-1, len(df.Embarked.value_counts()))
# specifies the parameters of our graphs
plt.title("Passengers per boarding location")
```

### 探索性可视化

本次竞赛的目标，是根据数据中的特征来预测一个人是否能够生还，例如：

- 乘客舱位等级（数据中的 `Pclass`）
- 性别（Sex）
- 年龄（Age）
- 票价（Fare）

我们来看看能否更好地理解谁生存、谁遇难。

首先，让我们画一张生还（Survived）和未生还人数对比的条形图。

```python
plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
df.Survived.value_counts().plot(kind='barh', color="blue", alpha=.65)
ax.set_ylim(-1, len(df.Survived.value_counts()))
plt.title("Survival Breakdown (1 = Survived, 0 = Died)")
```

### 现在我们再进一步挖掘结构

### 把上面的图按性别拆开来看

```python
fig = plt.figure(figsize=(18,6))

# create a plot of two subsets, male and female, of the survived variable.
# After we do that we call value_counts() so it can be easily plotted as a bar graph.
# 'barh' is just a horizontal bar graph
df_male = df.Survived[df.Sex == 'male'].value_counts().sort_index()
df_female = df.Survived[df.Sex == 'female'].value_counts().sort_index()

ax1 = fig.add_subplot(121)
df_male.plot(kind='barh',label='Male', alpha=0.55)
df_female.plot(kind='barh', color='#FA2379',label='Female', alpha=0.55)
plt.title("Who Survived? with respect to Gender, (raw value counts) "); plt.legend(loc='best')
ax1.set_ylim(-1, 2)

# adjust graph to display the proportions of survival by gender
ax2 = fig.add_subplot(122)
(df_male/float(df_male.sum())).plot(kind='barh',label='Male', alpha=0.55)
(df_female/float(df_female.sum())).plot(kind='barh', color='#FA2379',label='Female', alpha=0.55)
plt.title("Who Survived proportionally? with respect to Gender"); plt.legend(loc='best')

ax2.set_ylim(-1, 2)
```

从这里我们可以清楚地看到，按绝对数量来说，男性获救和遇难的人数都更多；但从**比例**上看，女性的生还率（约 25%）比男性（约 20%）更高。

#### 很好！不过我们还能再深入一点

我们能否通过加入 Pclass（舱位等级）进一步挖掘结构？这里我们把舱位分为低等级（3 等舱）与较高等级（1、2 等舱）。我们按性别和舱位等级一起拆分生还情况。

```python
fig = plt.figure(figsize=(18,4), dpi=1600)
alpha_level = 0.65

# building on the previous code, here we create an additional subset within the gender subset
# After we do that we call value_counts() so it can be easily plotted as a bar graph.
# This is repeated for each gender-class pair.
ax1 = fig.add_subplot(141)
female_highclass = df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts()
female_highclass.plot(kind='bar', label='female, highclass', color='#FA2479', alpha=alpha_level)
ax1.set_xticklabels(["Survived", "Died"], rotation=0)
ax1.set_xlim(-1, len(female_highclass))
plt.title("Who Survived? with respect to Gender and Class"); plt.legend(loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
female_lowclass = df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts()
female_lowclass.plot(kind='bar', label='female, low class', color='pink', alpha=alpha_level)
ax2.set_xticklabels(["Died","Survived"], rotation=0)
ax2.set_xlim(-1, len(female_lowclass))
plt.legend(loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)
male_lowclass = df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts()
male_lowclass.plot(kind='bar', label='male, low class',color='lightblue', alpha=alpha_level)
ax3.set_xticklabels(["Died","Survived"], rotation=0)
ax3.set_xlim(-1, len(male_lowclass))
plt.legend(loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
male_highclass = df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts()
male_highclass.plot(kind='bar', label='male, highclass', alpha=alpha_level, color='steelblue')
ax4.set_xticklabels(["Died","Survived"], rotation=0)
ax4.set_xlim(-1, len(male_highclass))
plt.legend(loc='best')
```

太棒了！现在我们对谁生还、谁遇难有了更多信息。凭借这种更深入的理解，我们可以构建更好、更有洞察力的模型。这是交互式数据分析中的典型流程：一开始从简单的关系入手，逐步理解最基础的模式，然后在不断发现更多数据特征的过程中逐步增加分析的复杂度。下面这段代码把这一流程的几个步骤并列展示出来：

```python
fig = plt.figure(figsize=(18,12), dpi=1600)
a = 0.65
# Step 1
ax1 = fig.add_subplot(341)
df.Survived.value_counts().plot(kind='bar', color="blue", alpha=a)
ax1.set_xlim(-1, len(df.Survived.value_counts()))
plt.title("Step. 1")

# Step 2
ax2 = fig.add_subplot(345)
df.Survived[df.Sex == 'male'].value_counts().plot(kind='bar',label='Male')
df.Survived[df.Sex == 'female'].value_counts().plot(kind='bar', color='#FA2379',label='Female')
ax2.set_xlim(-1, 2)
plt.title("Step. 2 \nWho Survived? with respect to Gender."); plt.legend(loc='best')

ax3 = fig.add_subplot(346)
(df.Survived[df.Sex == 'male'].value_counts()/float(df.Sex[df.Sex == 'male'].size)).plot(kind='bar',label='Male')
(df.Survived[df.Sex == 'female'].value_counts()/float(df.Sex[df.Sex == 'female'].size)).plot(kind='bar', color='#FA2379',label='Female')
ax3.set_xlim(-1,2)
plt.title("Who Survied proportionally?"); plt.legend(loc='best')

# Step 3
ax4 = fig.add_subplot(349)
female_highclass = df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts()
female_highclass.plot(kind='bar', label='female highclass', color='#FA2479', alpha=a)
ax4.set_xticklabels(["Survived", "Died"], rotation=0)
ax4.set_xlim(-1, len(female_highclass))
plt.title("Who Survived? with respect to Gender and Class"); plt.legend(loc='best')

ax5 = fig.add_subplot(3,4,10, sharey=ax1)
female_lowclass = df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts()
female_lowclass.plot(kind='bar', label='female, low class', color='pink', alpha=a)
ax5.set_xticklabels(["Died","Survived"], rotation=0)
ax5.set_xlim(-1, len(female_lowclass))
plt.legend(loc='best')

ax6 = fig.add_subplot(3,4,11, sharey=ax1)
male_lowclass = df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts()
male_lowclass.plot(kind='bar', label='male, low class',color='lightblue', alpha=a)
ax6.set_xticklabels(["Died","Survived"], rotation=0)
ax6.set_xlim(-1, len(male_lowclass))
plt.legend(loc='best')

ax7 = fig.add_subplot(3,4,12, sharey=ax1)
male_highclass = df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts()
male_highclass.plot(kind='bar', label='male highclass', alpha=a, color='steelblue')
ax7.set_xticklabels(["Died","Survived"], rotation=0)
ax7.set_xlim(-1, len(male_highclass))
plt.legend(loc='best')
```

我已经尽力让这些绘图代码做到可读、直观。如果你想更系统地了解如何使用 matplotlib 作图，可以看看这份非常漂亮的 Notebook：[链接在这里](http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb)。

现在我们已经对要预测的目标有了基本认识，接下来就真正开始预测。

## 监督式机器学习

#### 逻辑回归（Logistic Regression）

根据 Wikipedia 的解释：

> 在统计学中，逻辑回归（logistic regression 或 logit regression）是一类回归分析，用来预测一个**分类因变量**（该因变量只能取有限个取值，其数值大小本身没有意义，大小顺序是否有意义则不一定）的结果，基于一个或多个自变量。也就是说，它用于估计定性响应模型中参数的经验值。通过使用逻辑函数，将自变量（解释变量）的函数映射为概率，从而描述一次试验所有可能结果的概率。通常（也是本文中的情形），“逻辑回归”特指因变量为二元变量的问题——即只有两个类别；如果有两个以上类别，则称为多项逻辑回归（multinomial logistic regression）；如果多个类别是有序的，则称为有序逻辑回归（ordered logistic regression）。
>
> 逻辑回归通过使用概率分数作为因变量的预测值，来刻画一个分类因变量与一个或多个自变量之间的关系，这些自变量通常（但不一定）是连续变量。因此，它处理的问题与 probit 回归相同，并采用类似的技术。

#### 用我自己的话来概括一下：

这次竞赛希望我们预测一个**二元**结果，也就是说，它想知道某人是死亡（用 0 表示），还是存活（用 1 表示）。一个合理的起点是：计算每一条观测（也就是每一个乘客）成为 0 或 1 的概率。这样，我们就可以知道某人存活的概率，从而做出相对有依据的预测。如果这样做，我们会得到类似下面这样的结果：

![pred](https://raw.github.com/agconti/kaggle-titanic/master/images/calc_prob.png)

（*纵轴是某人存活的概率，横轴是乘客编号，从 1 到 891。*）

这样做得到的信息是有用的，不过它并不能直接告诉我们某人最终是生还是死，它只是告诉我们生还或遇难的概率。我们仍需要把这些概率转化为我们真正想要的二元决策。但怎么做呢？我们可以随意设定一个生还的概率阈值，比如 50%：只要存活概率高于 50% 就预测为生还。事实上，对于本数据集，这种做法表现还不错，能让你做出相当准确的预测。图形上大概会长这样：

![predwline](https://raw.github.com/agconti/kaggle-titanic/master/images/calc_prob_wline.png)

如果你像我一样喜欢“押注”，你可能并不愿意把一切都交给运气。把阈值设在 50% 就一定合理吗？也许 20% 或 80% 会更合适。显然，我们需要一种更精确的方法来确定这个阈值。谁来拯救我们？这就是**逻辑回归（Logistic Regression）**上场的地方。

逻辑回归做的事情和我们上面做的一样，只不过它会**用数学方法自动计算这个阈值**，也就是所谓的“决策边界（decision boundary）”。它会找出最适合训练数据的 cutoff，比如 50% 或 51.84%，从而尽可能准确地刻画训练数据。

下面的三个代码单元展示了创建逻辑回归模型、在数据上训练模型、以及检查模型表现的过程。

首先，我们先为逻辑回归定义一个公式。接着，我们用 patsy 的 dmatrices 函数创建一个对回归友好的 DataFrame，把公式中的分类变量转换为布尔值，并让模型知道我们输入的变量类型。然后实例化并拟合模型，最后打印模型表现的摘要。在最后一个单元中，我们把模型的预测值和真实值进行图形上的对比，同时查看模型残差，以检测是否还存在尚未捕捉到的结构。

```python
# model formula
# here the ~ sign is an = sign, and the features of our dataset
# are written as a formula to predict survived. The C() lets our
# regression know that those variables are categorical.
# Ref: http://patsy.readthedocs.org/en/latest/formulas.html
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked)'
# create a results dictionary to hold our regression results for easy analysis later
results = {}
# create a regression friendly dataframe using patsy's dmatrices function
y, x = dmatrices(formula, data=df, return_type='dataframe')

# instantiate our model
model = sm.Logit(y, x)

# fit our model to the training data
res = model.fit()

# save the result for outputing predictions later
results['Logit'] = [res, formula]
res.summary()
```

```python
# Plot Predictions Vs Actual
plt.figure(figsize=(18,4));
plt.subplot(121, axisbg="#DBDBDB")
# generate predictions from our fitted model
ypred = res.predict(x)
plt.plot(x.index, ypred, 'bo', x.index, y, 'mo', alpha=.25);
plt.grid(color='white', linestyle='dashed')
plt.title('Logit predictions, Blue: \nFitted/predicted values: Red');

# Residuals
ax2 = plt.subplot(122, axisbg="#DBDBDB")
plt.plot(res.resid_dev, 'r-')
plt.grid(color='white', linestyle='dashed')
ax2.set_xlim(-1, len(res.resid_dev))
plt.title('Logit Residuals');
```

## 这个模型表现得如何？

我们先从图形上看看模型的预测结果：

```python
fig = plt.figure(figsize=(18,9), dpi=1600)
a = .2

# Below are examples of more advanced plotting.
# If it looks strange check out the tutorial above.
fig.add_subplot(221, axisbg="#DBDBDB")
kde_res = KDEUnivariate(res.predict())
kde_res.fit()
plt.plot(kde_res.support, kde_res.density)
plt.fill_between(kde_res.support, kde_res.density, alpha=a)
plt.title("Distribution of our Predictions")

fig.add_subplot(222, axisbg="#DBDBDB")
plt.scatter(res.predict(), x['C(Sex)[T.male]'], alpha=a)
plt.grid(b=True, which='major', axis='x')
plt.xlabel("Predicted chance of survival")
plt.ylabel("Gender Bool")
plt.title("The Change of Survival Probability by Gender (1 = Male)")

fig.add_subplot(223, axisbg="#DBDBDB")
plt.scatter(res.predict(), x['C(Pclass)[T.3]'], alpha=a)
plt.xlabel("Predicted chance of survival")
plt.ylabel("Class Bool")
plt.grid(b=True, which='major', axis='x')
plt.title("The Change of Survival Probability by Lower Class (1 = 3rd Class)")

fig.add_subplot(224, axisbg="#DBDBDB")
plt.scatter(res.predict(), x.Age, alpha=a)
plt.grid(True, linewidth=0.15)
plt.title("The Change of Survival Probability by Age")
plt.xlabel("Predicted chance of survival")
plt.ylabel("Age")
```

### 使用模型预测测试集并导出结果到 Kaggle

```python
# Read the test data
test_data = pd.read_csv("data/test.csv")

# Examine our dataframe
test_data

# Add our dependent variable to our test data.
# (It is usually left blank by Kaggle because it is the value you are trying to predict.)
test_data['Survived'] = 1.23
```

我们封装好的结果字典：

```python
results
```

使用我们的模型对测试集进行预测，并把结果输出为 CSV 文件，便于提交 Kaggle：

```python
# Use your model to make prediction on our test set.
compared_resuts = ka.predict(test_data, results, 'Logit')
compared_resuts = Series(compared_resuts)  # convert our model to a series for easy output
# output and submit to kaggle
compared_resuts.to_csv("data/output/logitregres.csv")
```

### Kaggle 评分结果

- **RMSE = 0.77033**。这个结果还不错，等等等等。

```python
# Create an acceptable formula for our machine learning algorithms
formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'
```

### 支持向量机（SVM）

*“所以，呃……如果一条直线根本描述不了情况怎么办？”*

**Wikipedia：**

> 在机器学习中，支持向量机（Support Vector Machines，SVM）是一类带有相应学习算法的监督学习模型，用来分析数据并识别模式，用于分类和回归分析。最基本的 SVM 接收一组输入数据，对每个输入预测其属于两个类别中的哪一类，因而是一个非概率的二元线性分类器。给定一组训练样本，每个样本被标记为属于两个类别之一，SVM 训练算法会构建一个模型，把样本表示为空间中的点，并映射到高维空间，使得不同类别的样本被一个尽可能宽的间隔分开。新的样本随后被映射到同一空间，并根据它们落在间隔哪一侧来进行分类。
>
> 除了线性分类外，SVM 还可以通过所谓的核技巧（kernel trick）高效地实现非线性分类，即隐式地把输入映射到高维特征空间。

## 用我的话来解释

我们刚才实现的 Logit 模型很棒的一点，是它明确告诉我们应该把决策边界（也就是“生存阈值”）画在哪里。但如果你和我一样，可能会想：“所以，呃……如果一条直线根本不够呢？”线性决策边界尚可，但我们能不能做得更好？也许一个更复杂的决策边界——比如波浪线、圆形或者某种奇怪的多边形——会比直线更好地描述样本中的方差。

想象我们只根据年龄来预测生存。如果使用线性决策边界，那就意味着你每增加一岁，生还概率就线性增加或减少一个单位。但你很容易想到另一种可能的曲线：也许青壮年的生还概率最高，而非常年幼和非常年老的人生还概率都偏低。这就成了一个非常有趣的问题。然而我们的 Logit 模型只能学习线性决策边界，那怎么办呢？答案还是那个：\(MATH\)。

**答案是：**

我们可以把 logit 方程从表达线性关系：

\[ survived  = \beta_0 + \beta_1 pclass + \beta_2 sex + \beta_3 age + \beta_4 sibsp + \beta_5 parch + \beta_6 embarked \]

为了方便，可以简写成：

\[ y = x \]

变换到表达**非线性关系的线性形式**：

\[ \log(y) = \log(x) \]

这么做并没有“作弊”。Logit 模型**只**擅长建模线性关系，所以我们只是把一个非线性关系转化到另一条线性关系上。

一个容易理解的方式是看一个指数关系的图像，比如 \(x^3\)：

![x3](https://raw.github.com/agconti/kaggle-titanic/master/images/x3.png)

显然，这是一个非线性关系。如果我们把它直接拿去给 Logit 模型用（\(y = x^3\)），效果会很差。但如果我们对方程取对数：\(\log(y) = \log(x^3)\)，那么图像会变成下面这样：

![loglogx3](https://raw.github.com/agconti/kaggle-titanic/master/images/loglogx3.png)

看起来就非常接近线性了。

这种把模型变换到另一个数学空间、让其更易表达的做法，正是支持向量机背后所做的事情。它的数学原理并不简单，如果你感兴趣，可以戴上眼镜、慢慢读这份资料：[点这里](http://dustwell.com/PastWork/IntroToSVM.pdf)。下面的代码展示了实现一个 SVM 模型，并在 SVM 把我们的方程变换到三种不同数学空间后检查结果的过程。第一种是线性核，类似于我们的 Logit 模型；第二种是指数/多项式核；最后一种是 RBF（径向基）核。

```python
# set plotting parameters
plt.figure(figsize=(8,6))

# create a regression friendly data frame
y, x = dmatrices(formula_ml, data=df, return_type='matrix')

# select which features we would like to analyze
# try changing the selection here for different output.
# Choose : [2,3] - pretty sweet DBs [3,1] --standard DBs [7,3] -very cool DBs,
# [3,6] -- very long complex dbs, could take over an hour to calculate!
feature_1 = 2
feature_2 = 3

X = np.asarray(x)
X = X[:, [feature_1, feature_2]]

y = np.asarray(y)
# needs to be 1 dimensional so we flatten. it comes out of dmatrices with a shape.
y = y.flatten()

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)

X = X[order]
y = y[order].astype(np.float)

# do a cross validation
nighty_precent_of_sample = int(.9 * n_sample)
X_train = X[:nighty_precent_of_sample]
y_train = y[:nighty_precent_of_sample]
X_test = X[nighty_precent_of_sample:]
y_test = y[nighty_precent_of_sample:]

# create a list of the types of kernels we will use for your analysis
types_of_kernels = ['linear', 'rbf', 'poly']

# specify our color map for plotting the results
color_map = plt.cm.RdBu_r

# fit the model
for fig_num, kernel in enumerate(types_of_kernels):
    clf = svm.SVC(kernel=kernel, gamma=3)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=color_map)

    # circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=color_map)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
               levels=[-.5, 0, .5])

    plt.title(kernel)
    plt.show()
```

在这些图中，蓝色区域代表预测为“生还”，红色区域代表预测为“遇难”。看看线性核那张图：它把决策边界正好画在 50% 的地方！也就是说，我们刚才随口猜的 50% 阈值其实还挺不错的。正如你所见，后面那些非线性核生成的决策边界远比最开始的直线复杂得多。如果数据中确实存在更复杂的结构，这些更复杂的边界就有机会捕捉到，从而构建更强的预测模型。

接下来，你可以选择一种你喜欢的决策边界，调整下面的代码，并将结果提交给 Kaggle 看看效果如何。

```python
# Here you can output whichever result you would like by changing the Kernel and clf.predict lines
# Change kernel here to poly, rbf or linear
# adjusting the gamma level also changes the degree to which the model is fitted
clf = svm.SVC(kernel='poly', gamma=3).fit(X_train, y_train)

y, x = dmatrices(formula_ml, data=test_data, return_type='dataframe')

# Change the integer values within x.ix[:,[6,3]].dropna() to explore the relationships between other
# features. the ints are column positions. ie. [6,3] 6th column and the third column are evaluated.
res_svm = clf.predict(x.ix[:, [6,3]].dropna())

res_svm = DataFrame(res_svm, columns=['Survived'])
res_svm.to_csv("data/output/svm_poly_63_g10.csv")  # saves the results for you, change the name as you please.
```

### 随机森林（Random Forest）

“好吧，那如果画线 / 决策边界这一整套思路都没用呢？”

**Wikipedia，一如既往的“清晰”：**

> 随机森林是一种集成学习方法，用于分类（和回归），通过在训练时构建大量的决策树，并输出各个树结果中类别的众数作为最终分类结果。

**再用浅显一点的话说，以及它为什么与你有关：**

总会有持怀疑态度的人，你也许就是其中之一，对我们之前画的那些“漂亮的线”并不买账。那这里再给你一个选择：随机森林（Random Forest）。这是一种**非参数模型**，它完全抛弃了我们之前构建的那些公式，用的是纯粹的计算力再加上一点巧妙的统计观察，从数据中挖掘结构。

有一个关于口香糖球罐（gumball jar）的故事可以很好地说明随机森林背后的思想。我们几乎都玩过“猜罐子里有多少颗糖”的游戏，几乎没有人能猜得完全准确。但有趣的是，如果有足够多的人参与，虽然每个人的单独猜测都很糟糕，**所有人猜测的平均值**通常会离真实答案非常接近。很神奇吧？这个现象就是随机森林得以工作的那个“巧妙的统计观察”。

**它是怎么工作的？** 随机森林算法会在你的数据上随机抽取许多子样本，并在这些子样本上生成大量极其简单的模型来解释数据中的方差。这些模型就像每一次“猜糖数”的尝试，每一个单独看都很糟糕——真的很糟糕。但当我们对这些模型的预测取平均时，事情就变得有趣了：大多数糟糕的模型在平均过程中会互相抵消，它们的影响会被“平均到零”；而剩下的那部分模型，很可能恰好捕捉到了数据中的真实结构。

下面这个代码单元展示了如何实例化并拟合一个随机森林，生成预测结果，并对其进行评分。

```python
# import the machine learning library that holds the randomforest
import sklearn.ensemble as ske

# Create the random forest model and fit the model to our training data
y, x = dmatrices(formula_ml, data=df, return_type='dataframe')
# RandomForestClassifier expects a 1 dimensional NumPy array, so we convert
y = np.asarray(y).ravel()
# instantiate and fit our model
results_rf = ske.RandomForestClassifier(n_estimators=100).fit(x, y)

# Score the results
score = results_rf.score(x, y)
print "Mean accuracy of Random Forest Predictions on the data was: {0}".format(score)
```

我们的随机森林比“随手瞎猜”只好了一点点——也就是说，如果你随手给每个样本随机分配 0 或 1，只要样本足够多，平均表现几乎不会比这个结果差太多。看起来这一次，我们的随机森林并没有恰好“撞上”数据中的真实结构。


这些只是你可以使用的众多机器学习技术中的一部分。自己再多试试其他方法，向排行榜更高处冲刺吧！

准备好看一个更高级的分析示例了吗？可以看看这些 Notebook：

- [Kaggle Competition | Blue Book for Bulldozers Quantitative Model](http://nbviewer.ipython.org/github.com/agconti/AGC_BlueBook/master/BlueBook.ipynb#)
- [GOOG VS AAPL Correlation Arb](http://nbviewer.ipython.org/github.com/agconti/AGCTrading/master/GOOG%2520V.%2520AAPL%2520Correlation%2520Arb.ipynb)
- [US Dollar as a Vehicle Currency; an analysis through Italian Trade](https://github.com/agconti/US_Dollar_Vehicle_Currency)

#### 欢迎关注我：

- GitHub：[github.com/agconti](https://github.com/agconti)
- Twitter：[twitter.com/agconti](https://twitter.com/agconti)
