# CombinedANABotV1
 tool that converts PDF manuals into structured JSON metadata and generates embeddings compatible with the existing A.N.A Bot program. Contains both components of the project, creating and usage. 

You need to add a directory within your project named manuals with the PDF's inside.
You'd also need to download ollama, but there've been talks of being able to obtain an API key and possibly get a better 
AI for the program. 

You don't need the metadata, embeddings, and index files I've included. If you download the converters requirements and run the converter with a different set of PDF's in there, it will create those respective files :) 

I've updated the code and included both the program I used to create the metadata as well as the program that indexes and shows the user everything together. 
General steps for easy replication in my experience : 
1. Download pycharm and python and set up virtual environment
2. Download ollama, and pull whatever model is best for whatever machine you're running (ie; had to switch from llama3 to mistral when I uploaded to my laptop)
3. Transfer the requirements files and download the requirements onto the program. (ie; terminal run pip install -r bot_requirements.txt )
4. Upload both anabot and pdfconverter .py files
5. as detailed in the file, run
   python pdf_converter.py                    # Convert PDFs to JSON
    python pdf_converter.py --create-embeddings # Full pipeline: PDF -> Embeddings
6. Once all of the metadata and embeddings and extra files download, you can run the anabot program. Allow time for everything to fully load, as it may appear indexed but when you run the program it'll say there's not enough system storage,
when in reality it just needs some time to load everything properly.

Let me know if there's any questions! 
Here's my projects structure : 

<img width="656" height="644" alt="image" src="https://github.com/user-attachments/assets/a70e3c78-d415-4f3c-8a8a-7150b94f3283" />

