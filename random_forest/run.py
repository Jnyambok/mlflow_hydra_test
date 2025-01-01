import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from seaborn import sns
import argparse
import yaml
import os


def go(args):

    with open(args.model_config) as fp:
        model_config = yaml.safe_load(fp)
    
    titanic = sns.load_dataset('titanic')

    #preprocess the data: drop rows with missing values and convert categorical var to numerical
    titanic.dropna(subset=['age','embarked','deck'],inplace=True)
    titanic = pd.get_dummies(titanic,
                             columns=['sex','embarked','deck','class','who','embark_town','alive','alone'],
                             drop_first=True,
                             )
    
    #Define features and target
    X = titanic.drop('survived',axis=1)
    y = titanic['survived']

    #Split the data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    #Initialize the model
    #The double asterisks (**) in front of this dictionary unpacks it. This means that each key-value pair in the 'random_forest' dictionary is used as a keyword argument when creating the RandomForestClassifier object.
    rf = RandomForestClassifier(**model_config['random_forest'])

    #Train the model
    rf.fit(X_train,y_train)

    #Make predictions
    y_pred = rf.predict(X_test)

    #Calculate accuracy
    accuracy = accuracy_score(y_test,y_pred)

    print(f"\n\nAccuracy of RandomForest classifier on test set: {accuracy:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a YAML file containing the configuration for the random forest",
        required=True,
    )

    args = parser.parse_args()

    go(args)