import pandas as pd
import ast
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def prepare_cocktail_data(csv_file_path):
    # Load the dataset
    df = pd.read_csv(csv_file_path)

    # Parse the 'ingredients' column from string to list
    def parse_list(val):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    
    # Define the format for the embedding document
    def create_document_text(row):
        name = row.get('name', 'Unknown')
        category = row.get('category', 'Unknown')
        alcoholic = row.get('alcoholic', '')
        glass = row.get('glassType', '')
        instructions = row.get('instructions', '')
    
        # Parse ingredients and measures
        ingredients = parse_list(row.get('ingredients', '[]'))
        measures = parse_list(row.get('ingredientMeasures', '[]'))

        # Combine ingredients with its measure
        formatted_ingredients = []
        for i, ing in enumerate(ingredients):
            measure = measures[i] if i < len(measures) and measures[i] else ''
            measure = measure.strip()

            if measure:
                formatted_ingredients.append(f"{ing} {measure}")
            else:
                formatted_ingredients.append(ing)
        
        # Join ingredients with commas
        ingredients_text = ', '.join(formatted_ingredients)

        # Create the document text
        document_text = (
            f"Cocktail Name: {name} ({alcoholic})\n"
            f"Category: {category}\n"
            f"Glass Type: {glass}\n"
            f"Ingredients: {ingredients_text}\n"
            f"Instructions: {instructions}"
        )
        return document_text
    
    # Apply the function to every row
    df['embedding_document'] = df.apply(create_document_text, axis=1)
    return df

# Convert to json format for vector database ingestion
current_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.join(current_dir, 'documents')
csv_path = os.path.join(docs_dir, 'final_cocktails.csv')
json_path = os.path.join(docs_dir, 'cocktails.json')

df_processed = prepare_cocktail_data(csv_path)
print(df_processed['embedding_document'].iloc[0])
columns_to_keep = ['id','name', 'alcoholic', 'category', 'glassType', 'embedding_document']
df_processed[columns_to_keep].to_json(json_path, orient='records', indent=4)

# Implement JSONLoader with metadata extraction
def get_metadata(record:dict, metadata:dict) -> dict:
    metadata['id'] = record.get('id')
    metadata['name'] = record.get('name')
    metadata['alcoholic'] = record.get('alcoholic')
    metadata['category'] = record.get('category')
    metadata['glassType'] = record.get('glassType')
    return metadata

loader = JSONLoader(json_path, 
                    jq_schema='.[]', 
                    content_key='embedding_document',
                    json_lines=False,
                    text_content=True,
                    metadata_func=get_metadata)
data = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 3000,
    chunk_overlap = 300,
    length_function = len,
    add_start_index = True
)
chunks = text_splitter.split_documents(data)
print(f'Split {len(data)} cocktails (documents) into {len(chunks)} chunks.')

# Generate embeddings and store in ChromaDB
import chromadb
from chromadb.utils import embedding_functions

chroma_path = os.path.join(docs_dir, 'chroma_db')
default_ef = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path=chroma_path)

# Create a collection in ChromaDB
collection = chroma_client.get_or_create_collection(
    name = 'cocktails',
    embedding_function = default_ef
)

# Prepare the data for insertion into ChromaDB
ids = []
documents = []
metadatas = []

for doc in chunks:
    ids.append(f"doc_{doc.metadata.get('id')}_seq_{doc.metadata.get('seq_num',0)}")
    documents.append(doc.page_content)
    metadatas.append(doc.metadata)
print("Embedding and inserting documents into ChromaDB...")

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(f'Inserted {len(ids)} chunks into ChromaDB collection "cocktails".')

# Examine the stored data in ChromaDB
results = collection.get(
    limit=1,
    include=['embeddings','documents','metadatas']
)

print(results['metadatas'][0])
print(results['documents'][0])
embedding_vector = results['embeddings'][0]
print(f'total dimentions: {len(embedding_vector)}')
print(f'first 5 dimensions: {embedding_vector[:5]}')
