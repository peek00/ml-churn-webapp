from dotenv import dotenv_values
from pathlib import Path
import pymysql
import os
import pandas as pd
    
class DataETL:
    """
    Handles Extract and Joining of Data.
    """

    def __init__(self):
        """
        """
        self.__retrieve_secrets()
        pass

    def __retrieve_secrets(self):
        secrets = dotenv_values(".env")
        self.host =secrets['DB_HOST']
        self.name = secrets['DB_NAME']
        self.user = secrets['DB_USER']
        self.pwd = secrets['DB_PWD']

    def test_db_connection(self)->None:
        """
        Tries to open a connection to the database listed in host.
        Raises exception if fails.
        """
        connection = pymysql.connect(
            host = self.host,
            user = self.user,
            password = self.pwd,
            database = self.name
            )
        try:
            cursor = connection.cursor()
            cursor.close()
            connection.close()
            print(f"Successfully connected to database: {self.name}!")
        except Exception as e:
            print(e)
            print("Connection Failed!")

    def retrieve_tables(self, table_names:list=[
                                "account",
                                "account_usage",
                                "churn_status",
                                "city",
                                "customer"]
                        )->dict:
        mapping = {
            "account": "ACCOUNT_DF",
            "account_usage": "ACC_USAGE_DF",
            "churn_status": "CHURN_STATUS_DF",
            "city": "CITY_DF",
            "customer": "CUSTOMER_DF"
        }

        connection = pymysql.connect(
            host = self.host,
            user = self.user,
            password = self.pwd,
            database = self.name
            )
        
        database_obj = {}
        try:
            for table_name in table_names:
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql(query, connection)

                database_obj[table_name] = df
        finally:
            connection.close()
        for k,v in database_obj.items():
            assert len(v) != 0, f"DF created from {k} is empty!"
            setattr(self, mapping[k], v)
            
        return database_obj

    def join_tables(self):
        """
        Join all tables in the database.
        """
        # Joining customer and account tables on customer_id
        try:
            DF = pd.merge(self.CUSTOMER_DF, self.ACCOUNT_DF, on='customer_id', how='inner')
            # Joining df with account_usage_df on account_id
            DF = pd.merge(DF, self.ACC_USAGE_DF, on='account_id', how='inner')
            # Joining df with churn_status_df on customer_id
            DF = pd.merge(DF, self.CHURN_STATUS_DF, on='customer_id', how='inner')
            # Joining df with city_df on area_id
            DF = pd.merge(DF, self.CITY_DF, on='zip_code', how='inner')
            self.df = DF
            print("DF Merged Successfully!")
        except Exception as e:
            print(e)
            print("Error while merging DFs!")

    def use_local_data(self, folder:Path)->None:
        """
        Use local data instead of connecting to database, only for experimentaitons.
        ---
        Arguments:
        folder: Path to folder containing all the csv files.
        """
        self.ACCOUNT_DF = pd.read_csv(os.path.join(folder, "1_account.csv"))
        self.ACC_USAGE_DF = pd.read_csv(os.path.join(folder, "2_account_usage.csv"))
        self.CHURN_STATUS_DF = pd.read_csv(os.path.join(folder, "3_churn_status.csv"))
        self.CITY_DF = pd.read_csv(os.path.join(folder, "4_city.csv"))
        self.CUSTOMER_DF = pd.read_csv(os.path.join(folder, "5_customer.csv"))

        assert not self.ACCOUNT_DF.empty , "Account DF is None!"
        assert not self.ACC_USAGE_DF.empty , "Account Usage DF is None!"
        assert not self.CHURN_STATUS_DF.empty , "Churn Status DF is None!"
        assert not self.CITY_DF.empty , "City DF is None!"
        assert not self.CUSTOMER_DF.empty , "Customer DF is None!"

    def get_df(self)->pd.DataFrame:
        """
        Returns the joined dataframe.
        """
        return self.df
    
   
        

if __name__ == "__main__":
    etl = DataETL()
    # etl.test_db_connection()
    etl.retrieve_tables()
    # etl.use_local_data("research/data_given/")
    etl.join_tables()
    print(etl.df.head())