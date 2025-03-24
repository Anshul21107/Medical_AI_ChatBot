# 🏥 Medical AI ChatBot

Welcome to **Medical AI ChatBot**, an intelligent assistant designed to provide medical insights and answer health-related queries. By leveraging advanced **Retrieval-Augmented Generation (RAG)** and NLP models, it ensures accurate and context-aware responses.

---

## 🚀 Features

🔹 **Conversational AI**: Engage in interactive, real-time medical discussions  
🔹 **Document Processed**: Use medical PDFs to enhance chatbot knowledge  
🔹 **RAG Mechanism**: Combines retrieval with generative models for precision  
🔹 **Streamlit Interface**: User-friendly and easy to navigate  

---

## 🧠 Model & Dataset

🔹 **LLM Model**: [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)  
🔹 **Embedding Model**: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
🔹 **Dataset**:  
   - *Gale Encyclopedia of Medicine Vol. 2 (N-S).pdf* [(Download Link)](https://github.com/Anshul21107/Medical_AI_ChatBot/blob/main/Data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf)  
   - *Cancer and Cure: A Critical Analysis* [(Download Link)](https://github.com/Anshul21107/Medical_AI_ChatBot/blob/main/Data/cancer_and_cure__a_critical_analysis.27.pdf)  
   - *Medical Oncology Handbook (June 2020 Edition)* [(Download Link)](https://github.com/Anshul21107/Medical_AI_ChatBot/blob/main/Data/medical_oncology_handbook_june_2020_edition.pdf)  

---

## 🛠️ Tech Stack

🔹 **Programming Language**: Python 🐍  
🔹 **Framework**: Streamlit 🎨  
🔹 **ML Libraries**: LangChain, HuggingFace Transformers, FAISS  

---

## 🏠 Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Anshul21107/Medical_AI_ChatBot.git
cd Medical_AI_ChatBot
```

### 2️⃣ Install Dependencies
Ensure Python is installed, then run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare the Knowledge Base
Place medical PDFs in the `data/` directory and process the documents:
```bash
python create-memory.py
```

### 4️⃣ Run RAG-based Retrieval
```bash
python rag.py
```

### 5️⃣ Launch the Chatbot
```bash
streamlit run medical-bot.py
```
---

## 📸 Screenshots

<p align="center">
  <img src="https://github.com/Anshul21107/Medical_AI_ChatBot/blob/main/Screenshot/Screenshot%202025-03-24%20115610.png" width="600">
</p>

<p align="center">
  <img src="https://github.com/Anshul21107/Medical_AI_ChatBot/blob/main/Screenshot/Screenshot%202025-03-24%20115953.png" width="600">
</p>


---
## 👤 Author

Developed by **Anshul Katiyar** 
