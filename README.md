# READme file NLP_for_Creative

# Breastfeeding support chatbot

## Overview

This project leverages data collected from the La Leche League UK website to develop a chatbot that helps users access information, regarding breastfeeding and parenting support, by utilizing data from La Leche League blog post resources. The project aims to provide 24/7 access to practical information and advice on breastfeeding, answering common questions and offering tips for new mothers, as an alternative to the online chat service with breastfeeding councillors.

**Data Description**

The chatbot is based on datasets extracted from articles in the breastfeeding information archives on the La Leche League UK website.

**Data Files**

Below is the data structure for **separate_lll_data** as well as **combined_lll_data.csv**

| **Column Name** | **Description** |
| --- | --- |
| **Post Title** | The title of the breastfeeding article. |
| **URL** | The web address (link) where the article can be accessed. |
| **Content** | The text data of the blog post article |

Below is the data structure for **preprocessed_lll_data.csv**

| **Column Name** | **Description** |
| --- | --- |
| **Paragraph** | Organised article text data split into paragraphs  |
| **Source** | The web address (link) where the article can be accessed. |
| **Title** | The title of the breastfeeding article. |

**Other Files**
**webscraping.ipynb** notebook showing how I extracted data from the website
**combining data.ipynb** notebook showing how I combined dataframes for each webpage into one
**pre_processing.ipynb** notebook showing how I organised the data into paragraph chunks
**initial_transformer.ipynb** notebook of first chatbot
**final_transformer.ipynb** notebook of enhanced chatbot
**CLI_source_code.py** final chatbot code for CLI

### **Project Steps**

1. **Data Collection & Preprocessing**
    - Ethically webscrape data from website pages and extract relevant text content.
    - Combine the individual data frames from each webpage into one structured data frame.
    - Preprocess the La Leche League articles by segmenting them into manageable paragraphs.
2. **Embedding Creation**
    - Generate vector embeddings for each paragraph using the Hugging Face pipeline **Sentence-BERT (`all-MiniLM-L6-v2`)**.
    - These embeddings allow for efficient semantic similarity comparisons.
3. **Intent Detection (`detect_intent`)**
    - Before retrieving relevant content, the system detects the user’s intent to understand the nature of the query.
    - This helps classify questions into categories such as:
        - **Information-seeking** (e.g., “What are the benefits of breastfeeding?”)
        - **Instructional** (e.g., “How do I increase my milk supply?”)
        - **Emotional support** (e.g., “I feel discouraged about breastfeeding, what should I do?”)
    - Intent detection improves response accuracy by directing the query to the most relevant content.
4. **Retrieval Process**
    - The user’s question is converted into an embedding.
    - The system searches for the most semantically similar paragraphs by comparing embeddings.
    - The top matching paragraphs are retrieved based on their relevance scores.
5. **Answer Generation**
    - The most relevant retrieved paragraphs are combined into a single context.
    - A **Question-Answering (QA) model** (**`deepset/roberta-base-squad2`**) extracts the specific answer from this context.
    - If the response is too long or spans multiple paragraphs, it is condensed using the **Hugging Face Summarization pipeline** (**`facebook/bart-large-cnn`**).
    - The final answer is presented in a concise and readable format, ensuring clarity.
6. **Adding Related Topic Suggestions**
    - To enhance user experience, the system suggests related topics based on the retrieved content.

**Technologies Used**

- **BeautifulSoup, requests & random**: for web scraping
- **Pandas, global & os :** For combining and preprocessing
- **Hugging Face Transformers**
- **Scikit-learn, numpy, pytorch & sentence transformers**


**Future Directions**

Future enhancements may include testing and finetuning model, making the chatbot better at understanding the nuances in the breastfeeding support queries. Then exploring frameworks like **Rasa** would be beneficial, as it provides tools for context handling, slot filling, and dialogue management out of the box, for a more advanced, production-ready solution.

**Acknowledgements**

Thanks to LalecheLeagueUK for the information resources that made this project possible. This project is for educational and self-improvement purposes.

##