
if __name__ == "__main__":

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from matcher import search
    import pandas as pd
    from scipy import sparse
    import joblib
    from text_processor import text_processing
    import uvicorn    
    from vector_store import storing, pinecone_search
    from index_creator import create_index
    

    antique = {
        "vectorizer": joblib.load("C:/Users/HP/Desktop/IR_FILES/antique/vectorizer.pkl"),
        "tfidf_matrix": sparse.load_npz("C:/Users/HP/Desktop/IR_FILES/antique/tfidf_matrix.npz"),
        "dataset": pd.read_csv("C:/Users/HP/Desktop/IR_FILES/antique/dataset.csv", usecols=[0, 1]),
    }

    clinical = {
        "vectorizer": joblib.load("C:/Users/HP/Desktop/IR_FILES/clinical/vectorizer.pkl"),
        "tfidf_matrix": sparse.load_npz("C:/Users/HP/Desktop/IR_FILES/clinical/tfidf_matrix.npz"),
        "dataset": pd.read_csv("C:/Users/HP/Desktop/IR_FILES/clinical/dataset.csv", usecols=[0, 1]),
    }

    dataset = None
    vectorizer = None    


    app = FastAPI()

    origins = ["http://localhost:5173"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    class TextProcessingRequest(BaseModel):
        text: str


    class IndexingRequest(BaseModel):
        dataset_path: str


    class SearchingRequest(BaseModel):        
        query: str
        dataset: str    


    class VectorStoringRequest(BaseModel):
        api_key: str
        tfidf_matrix_path: str
        index_name: str


    class PineconeSearchingRequest(BaseModel):
        api_key: str
        vectorizer_path: str
        dataset_path: str
        index_name: str
        query: str
    

    @app.post("/text-processing")
    def process_text(request: TextProcessingRequest):
        try:
            processed_text = text_processing(request.text)
            return {"processed_text": processed_text}
        except Exception as e:        
            return {"message": "something went wrong while processing", "details": str(e)}
        

    @app.post("/indexing")
    def indexing(request: IndexingRequest):
        try:
            create_index(request.dataset_path)
            return {"message": "index created successfully"}
        except Exception as e:
            return {"message": "something went wrong while creating index", "details": str}


    @app.post("/matching")
    def matching(request: SearchingRequest):
        try:            
            if request.dataset == "antique":
                top_ids, top_docs = search(antique.get("vectorizer"), antique.get("tfidf_matrix"), antique.get("dataset"), request.query)
            else:
                top_ids, top_docs = search(clinical.get("vectorizer"), clinical.get("tfidf_matrix"), clinical.get("dataset"), request.query)
            return {"top_ids": top_ids, "top_docs": top_docs}
        except FileNotFoundError as e:
            return {"message": "file not found", "details": str(e)}
        except Exception as e:
            return {"message": "something went wrong while searching", "details": str(e)}                    


    @app.post("/vector-storing")
    def vector_storing(request: VectorStoringRequest):
        try:
            tfidf_matrix = sparse.load_npz(request.tfidf_matrix_path)
            storing(request.api_key, tfidf_matrix, request.index_name)
            return {"message": "vectors stored sucessfully"}
        except Exception as e:
            return {"message": "something went wrong while storing vectors", "details": str(e)}


    @app.post("/pinecone-search")
    def pinecone_matching(request: PineconeSearchingRequest):
        try:
            global vectorizer, dataset
            if vectorizer is None:
                vectorizer = joblib.load(request.vectorizer_path)
            if dataset is None:
                dataset = pd.read_csv(request.dataset_path)
            top_ids, top_docs = pinecone_search(vectorizer, dataset, request.api_key, request.index_name, request.query)
            return {"top_ids": top_ids, "top_docs": top_docs}
        except Exception as e:
            return {"message": "something went wrong while searching", "details": str(e)}



    uvicorn.run(app, host="127.0.0.1", port=8001)