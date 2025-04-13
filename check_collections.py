from statsModelsPredict.src.db_config.mongodb_handler import MongoDBHandler
handler = MongoDBHandler()
handler.connect()
collections = ["preprocessed", "preprocessed_3", "preprocessed_etc"]
for coll in collections:
    sample = handler.db[coll].find_one()
    print(f"\n컬렉션 {coll} 컬럼 목록:")
    print(list(sample.keys()) if sample else "No data")
handler.close()
