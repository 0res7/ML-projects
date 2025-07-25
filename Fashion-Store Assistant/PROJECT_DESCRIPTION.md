
# Project Description: Fashion-Store Assistant

## Objective
The main objective of this project is to develop a proof-of-concept application that can intelligently process email order requests and customer inquiries for a fashion store. The system should be able to accurately categorize emails as either product inquiries or order requests and generate appropriate responses using the product catalog information and current stock status.

## Dataset Used
The project uses a Google Spreadsheet as its data source, which contains two main tabs:

- **Products**: A list of products with fields including product ID, name, category, stock amount, detailed description, and season.
- **Emails**: A sequential list of emails with fields such as email ID, subject, and body.

## ML Pipeline
The machine learning pipeline in this project is as follows:

1.  **Data Loading and Preprocessing**: The data is loaded from the Google Spreadsheet into pandas DataFrames. The product catalog and email data are preprocessed to prepare them for use in the machine learning models.

2.  **Email Classification**: A Large Language Model (LLM) is used to classify the emails into one of two categories: "Order Request" or "Product Inquiry". The model is fine-tuned on the email dataset to learn the distinguishing features of each category.

3.  **Information Extraction**: For order requests, the LLM is used to extract key information from the email body, such as the product name, size, and quantity.

4.  **Response Generation**: The system generates an appropriate response based on the email category and the extracted information. For product inquiries, the system uses Retrieval-Augmented Generation (RAG) and a vector store to retrieve relevant information from the product catalog and generate a helpful response. For order requests, the system checks the stock status of the requested product and generates a response indicating whether the order can be fulfilled.

## Technical Stack
- **Programming Language**: Python
- **Libraries**:
    - **Pandas**: For data manipulation and analysis.
    - **NumPy**: For numerical operations.
    - **Scikit-learn**: For implementing machine learning algorithms, preprocessing, and evaluation.
    - **OpenAI API**: For accessing the GPT-4o model.
    - **LangChain**: For building applications with LLMs.
    - **ChromaDB**: For creating and managing the vector store.

## Key Insights
- The use of a powerful LLM like GPT-4o allows the system to handle complex tasks, such as email classification and information extraction, with high accuracy.
- The RAG and vector store techniques are effective for retrieving relevant information from the product catalog and generating informative responses to product inquiries.
- The system is able to automate the process of handling customer emails, which can save time and improve efficiency for the fashion store.
