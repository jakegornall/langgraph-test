import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from react_agent.ChaseAzureOpenAI import get_access_token_headers, getModel
from react_agent.models import VectorChunks
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

docs_folder = "./docs"
openai_embedding_bucket_size = 166

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    default_headers=get_access_token_headers()
)

model = getModel()
structured_llm = model.with_structured_output(VectorChunks)

vector_store = Chroma(embedding_function=embeddings)
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def process_document(doc, framework):
    prompt = (
        "You are tasked with reading the following document and generating a list of structured text chunks. "
        "Each chunk should include the name of the package/library or tool it is for, the method or component name, "
        "and if necessary, an example of code for how to import and use the component or library or method. "
        "Chunks can be long or short depending on what is needed to cover the topic. "
        "Example Chunk: \n"
        "Package: @octagon/analytics\n"
        "Title: How to Install and UseOctagon Analytics\n"
        "Description: First, install the package via npm, then import the event you would like to fire and call the function with the parameters the event requires. \n"
        "```bash\n"
        "npm install @octagon/analytics\n"
        "```\n"
        "Usage: \n"
        "```typescript\n"
        "import { screenEvent } from '@octagon/analytics';\n"
        "\n"
        "screenEvent({\n"
        "  screenId: 'myApp/myProduct/myFeature/myPageOne'\n"
        "});\n"
        "```"
        "Another Example Chunk: \n"
        "Package: @mds/web-ui-button\n"
        "Title: MDS Web UI Button Installation and Usage\n"
        "Description: The MDS Web UI Button is a React component with the following props: \n"
        "Props: \n"
        "  variant: 'primary' | 'secondary' | 'tertiary'\n"
        "  href: string\n"
        "  text: string\n"
        "Usage: \n"
        "```typescript\n"
        "import { Button } from '@mds/web-ui-button';\n"
        "\n"
        "<Button\n"
        "  variant='primary'\n"
        "  href='https://www.chase.com/'\n"
        "  text='Submit'\n"
        "/>\n"
        "```"
        "\n"
    )
    while True:
        try:
            print("Processing document")
            response = structured_llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Please read the following document and generate a list of structured text chunks:\n" + doc.page_content)])
            chunks = response.chunks
            print(f"Received {len(chunks)} chunks")

            metadataValues = {"package_name": [], "framework": []}

            # load metadata file if it exists
            if os.path.exists("metadata.json"):
                with open('metadata.json') as json_file:
                    metadataValues = json.load(json_file)

            for chunk in chunks:
                metadataValues["package_name"].append(chunk.package_name)
                metadataValues["framework"].append(framework)
                document = Document(page_content=chunk.content, metadata={"package_name": chunk.package_name, "framework": framework})
                
                # Add each document to the vector store immediately
                vector_store.add_documents(documents=[document])
                print(f"Added {chunk.package_name} to vector store")
                
                metadataValues["package_name"].append(chunk.package_name)
                metadataValues["framework"].append(framework)

                ## dedupe metadata
                metadataValues = {k: list(set(v)) for k, v in metadataValues.items()}

                with open('metadata.json', 'w') as json_file:
                    json.dump(metadataValues, json_file)

            return  # No need to return documents since they are added immediately
        except Exception as e:
            print(f"Error processing document: {e}. Retrying in 3 seconds...")
            time.sleep(3)

def load_docs(doc_list, framework):
    print("load_docs(", str(len(doc_list)), " doc(s))", sep='')
    with ThreadPoolExecutor() as executor:
        future_to_doc = {executor.submit(process_document, doc, framework): doc for doc in doc_list}
        for future in as_completed(future_to_doc):
            try:
                future.result()  # We don't need to collect results since they are added immediately
            except Exception as e:
                print(f"Document processing failed: {e}")

print("Done.")

if __name__ == "__main__":
    print("Loading MDS docs...")
    path = docs_folder + '/mds'
    text_loader_kwargs = {'encoding': 'ISO-8859-1'}
    loader = DirectoryLoader(path, glob='**/*.txt', loader_cls=TextLoader, use_multithreading=True, show_progress=True, loader_kwargs=text_loader_kwargs)
    mds_docs = loader.load()
    load_docs(mds_docs, "MDS")

    print("Done.")

    retrieved_docs = vector_store.similarity_search(query="What is the MDS component for a button and how do I use it?", k=50, filter={"package_name": "@mds/web-ui-button"})
    print(retrieved_docs)