# PDF Chatbot with Streamlit and LLMs

## Overview
The PDF Chatbot is a powerful application designed to facilitate interactive discussions directly with the content of PDF documents. Built using Streamlit and leveraging cutting-edge language models from Hugging Face and LangChain, this tool parses any uploaded PDF document and allows users to engage in a question-and-answer session based on the document's content. This project is ideal for anyone looking to extract information from PDFs through natural language queries, making it an invaluable resource for researchers, students, professionals, and anyone dealing with document-based data.

## Features
- **PDF Reading and Text Extraction:** Utilize PyPDF2 to read PDF files and extract text seamlessly.
- **Interactive Q&A:** Ask questions and receive answers based on the PDF's content, powered by the latest transformer models from Hugging Face.
- **Intelligent Search:** Implements a FAISS vector store for efficient similarity search within the document text to find relevant content quickly.
- **Conversation History:** Mimics a chat-like interface where both questions and responses are logged for easy reference during the session.
- **Responsive UI:** Built using Streamlit, the app offers a clean and responsive user interface that is intuitive to use.

## Technologies Used
- **Streamlit:** An open-source app framework for Machine Learning and Data Science projects.
- **Hugging Face Transformers:** Provides state-of-the-art machine learning models.
- **PyPDF2:** A Pure-Python library built as a PDF toolkit.
- **FAISS:** A library for efficient similarity search and clustering of dense vectors.
- **LangChain:** A toolkit for building language model applications quickly and with less code.

## Setup and Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/DenisMahajan/ChatwithPDF.git
   cd ChatwithPDF
   ```

2. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables:**
   Create a `.env` file in the project root and populate it with necessary values:
   ```
   API_KEY=your_huggingface_api_key
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Usage
Upload a PDF through the Streamlit interface and start asking questions directly related to the document's content. The application will display responses and maintain a dynamic conversation history for reference.

---
