import os
os.environ["SERPER_API_KEY"] = "bb97567bf9dad276d35e1c5c7446fd8501451f81"

from langchain.utilities import GoogleSerperAPIWrapper

search = GoogleSerperAPIWrapper()


