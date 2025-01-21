import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
import dagshub



# ml flow set tracking 
mlflow.set_tracking_uri("https://dagshub.com/AMR-ITH/mlflow-dagshub-demo.mlflow")


dagshub.init(repo_owner='AMR-ITH', repo_name='mlflow-dagshub-demo', mlflow=True)



# load the iris data set

iris = load_iris()
X = iris.data
y = iris.target

# split the dataset into training and training sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Define the parameter for the random forest 
max_depth=20

# apply ml flow

mlflow.set_experiment('iris-dt')

with mlflow.start_run():

    dt = DecisionTreeClassifier(max_depth=max_depth)

    dt.fit(X_train,y_train)

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)

    mlflow.log_param('max_depth',max_depth)

    # mlflow.log_param('n_estimators',n_estimators)

    
    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')

    # Save the confusion matrix
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(dt,"Decession Tree")

    mlflow.set_tag('authur','amr')
    mlflow.set_tag('model','decision tree')

    print('accu',accuracy)
