# Importando bibliotecas

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Teste de imports. Se printar ok sem erros, os imports foram realizados com sucesso
print('Ok')

# Carregando DataSet
url = "DataSet/iris-dataset.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Dimensão do DataSet
# Deve printar (150, 5)
print(dataset.shape)

# Informações do DataSet(Peek)
# Pega as primeiras 20 linhas do dataset
print(dataset.head(20))

# Informações estatísticas
# Mostra informações estatísticas do dataset
print(dataset.describe())

# Distribuição de Classe
# Mostra a quantidade de linhas por class de flor
print(dataset.groupby('class').size())


dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

dataset.hist()
pyplot.show()

scatter_matrix(dataset)
pyplot.show()

# Criar um dataset de validação
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)
