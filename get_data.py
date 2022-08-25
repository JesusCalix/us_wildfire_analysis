# metadata = MetaData()
# fires = Table('Fires', metadata, autoload=True, autoload_with=engine)
# # print(session.query(fires).count())
# query = select([fires])
# result_proxy = connection.execute(query)
# result_set = result_proxy.fetchall()
# print(result_set[:2])

# columns = fires.c
# for c in columns:
#     print (c.name, c.type)

from kaggle.api.kaggle_api_extended import KaggleApi

def get_sqlite_db():
    try:
        print("Authenticating with Kaggle...")
    
        api = KaggleApi()
        api.authenticate()
        print("Downloading kaggle dataset...")
        api.dataset_download_files(dataset='rtatman/188-million-us-wildfires',
                                    path = './data/',
                                    unzip=True)

        print("Data has been downloaded and extracted into the destination.")
    except Exception as e: 
        print(e)

def main():
    get_sqlite_db()

if __name__ == "__main__":
    main()