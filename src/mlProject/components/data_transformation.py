import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


# Define the DataFrameSelector class
class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select specified columns from a pandas DataFrame.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        # No fitting needed, so return self
        return self

    def transform(self, X):
        # Select the specified columns from the DataFrame
        return X[self.columns].values


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    ## --------------------- Data Processing ---------------------------- ##

    def data_transformation(self):
        """
        Scale numerical features in the dataset using StandardScaler.
        """

        logger.info("Loading dataset...")

        df = pd.read_csv(self.config.data_path)

        df.drop(columns=["RowNumber","CustomerId","Surname"], inplace=True)
        
        ## To features and target
        X = df.drop(columns=['Exited'], axis=1)
        y = df['Exited']

        ## Split to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y)

        logger.info("Starting data processing..")

        ## Slice the lists
        num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
        categ_cols = ['Gender', 'Geography']

        ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))

        ## For Numerical
        num_pipeline = Pipeline(steps=[
                                ('selector', DataFrameSelector(num_cols)),
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())
                            ])

        ## For Categorical
        categ_pipeline = Pipeline(steps=[
                                ('selector', DataFrameSelector(categ_cols)),
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('ohe', OneHotEncoder(drop='first', sparse_output=False))
                         ])

        ## For ready cols
        ready_pipeline = Pipeline(steps=[
                                ('selector', DataFrameSelector(ready_cols)),
                                ('imputer', SimpleImputer(strategy='most_frequent'))
                            ])

        ## combine all
        Transformation_pipline = ColumnTransformer(transformers=[
                                            ('numerical', num_pipeline, num_cols),
                                            ('categorical', categ_pipeline, categ_cols),
                                            ('ready', ready_pipeline, ready_cols)
                                     ])

        ## apply
        Transformation_pipline.fit(X_train)

        ## As I did OHE, The column number may vary
        out_categ_cols = Transformation_pipline.named_transformers_['categorical'].named_steps['ohe'].get_feature_names_out(categ_cols)

        X_train_final = pd.DataFrame(Transformation_pipline.transform(X_train), columns=num_cols + list(out_categ_cols) + ready_cols)
        X_test_final = pd.DataFrame(Transformation_pipline.transform(X_test), columns=num_cols + list(out_categ_cols) + ready_cols)

        train = pd.concat([X_train_final.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test = pd.concat([X_test_final.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        # Save transformed train and test data
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        
        try:
            joblib.dump(Transformation_pipline, os.path.join(self.config.root_dir, self.config.transformation_pipline_name))
            logger.info(f"Transformation_pipline saved successfully at {os.path.join(self.config.root_dir, self.config.transformation_pipline_name)}")
        except Exception as e:
            logger.info(f"Error saving Transformation_pipline: {e}")

        logger.info("Feature scaling completed.")