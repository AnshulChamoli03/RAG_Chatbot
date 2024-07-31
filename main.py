import os
import sys
import getpass
from tkinter import *
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

folder_path = "C:/Users/anshul.chamoli/Desktop/data_retieve"


# Load documents from a folder
def readDoc():
    documents = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith('.docx') or file.endswith('.doc'):
            loader = Docx2txtLoader(file_path)
        elif file.endswith('.txt'):
            loader = TextLoader(file_path)
        # elif file.endswith('.xlsx'):
        #     loader = UnstructuredExcelLoader(file_path)
        documents.extend(loader.load())

    # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(documents)
    return documents


documents = readDoc()
# Convert document chunks to embeddings and save to vector store
vectordb = FAISS.from_documents(documents, embedding=OpenAIEmbeddings())

# Create conversational QA chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

# Tkinter GUI setup
root = Tk()
root.title("Doc Chatbot")  # Set window title

# Set window size and position
window_width = 1200
window_height = 900

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)

root.geometry(f"{window_width}x{window_height}")

# New color palette
background_color = "#1e1e1e"  # Dark background
foreground_color = "#e1e1e1"  # Light gray text
button_color = "#3b3b3b"  # Dark button
button_text_color = "#ffffff"  # White text on buttons
highlight_color = "#5c5c5c"  # Highlight color for frames and borders
user_msg_color = "#2e2e2e"  # Darker background for user messages
bot_msg_color = "#4a4a4a"  # Slightly lighter background for bot messages

# Title label
title_label = Label(root, text="Doc Chatbot", font=("Arial", 24), bg=background_color, fg=foreground_color)
title_label.pack(pady=20)

# Frame for displaying chat messages with border
chat_frame = Frame(root, width=700, height=75 * 7, bg=highlight_color, bd=1, relief=SOLID)  # Highlighted border
chat_frame.pack_propagate(False)
chat_frame.place(relx=0.5, rely=0.5, anchor=CENTER)

# Canvas for chat frame
chat_canvas = Canvas(chat_frame, bg=highlight_color, bd=1, relief=SOLID)
chat_canvas.pack(side=LEFT, fill=BOTH, expand=True)

# Scrollbar for chat frame
chat_scrollbar = Scrollbar(chat_frame, orient=VERTICAL, command=chat_canvas.yview, bg=background_color)
chat_scrollbar.pack(side=RIGHT, fill=Y)

# Configure canvas
chat_canvas.configure(yscrollcommand=chat_scrollbar.set)
chat_canvas.bind('<Configure>', lambda e: chat_canvas.configure(scrollregion=chat_canvas.bbox("all")))

# Frame inside canvas
inner_chat_frame = Frame(chat_canvas, bg=highlight_color)
chat_canvas.create_window((0, 0), window=inner_chat_frame, anchor="nw")

# User input field
user_input = Entry(root, width=90, bg=background_color, fg=foreground_color, insertbackground=foreground_color)
user_input.place(relx=0.5, rely=0.9, anchor=S)


# Function to ask question and display answer
def ask_question():
    populate_listbox()
    question = user_input.get().strip()
    if question:
        result = pdf_qa.invoke({"question": question, "chat_history": []})
        answer = result["answer"]
        display_message(question, "user")
        display_message(answer, "bot")
    user_input.delete(0, END)  # Clear the entry field after asking


# Ask button
ask_button = Button(root, text="Ask", width=10, command=ask_question, bg=button_color, fg=button_text_color)
ask_button.place(relx=0.85, rely=0.9, anchor=S)

# Frame for listbox and scrollbar
listbox_frame = Frame(root, bg=background_color)
listbox_frame.place(relx=0, rely=0.2)

# Listbox to display files
listbox = Listbox(listbox_frame, width=25, height=20, bg=background_color, fg=foreground_color)
listbox.pack(side=LEFT, fill=BOTH)

# Scrollbar for listbox
scrollbar = Scrollbar(listbox_frame, bg=background_color)
scrollbar.pack(side=RIGHT, fill=Y)

# Attach scrollbar to listbox
listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)


# Function to populate listbox with files
def populate_listbox():
    listbox.delete(0, END)
    readDoc()
    for file in os.listdir(folder_path):
        listbox.insert(END, file)


# Populate the listbox initially
populate_listbox()

# Handle Enter key press
root.bind('<Return>', lambda event=None: ask_question())


# Display messages in the chat window
def display_message(message, sender):
    if sender == "user":
        color = user_msg_color  # Darker background for user messages
        justify = "right"
        message_label = Label(inner_chat_frame, text=message, bg=color, fg=foreground_color, wraplength=650,
                              justify=justify, padx=10, pady=5, anchor="w", width=93)
        message_label.pack(side=TOP, fill=X)
    else:
        color = bot_msg_color  # Slightly lighter background for bot messages
        justify = "left"
        message_label = Label(inner_chat_frame, text=message, bg=color, fg=foreground_color, wraplength=650,
                              justify=justify, padx=10, pady=5, anchor="w", width=93)
        message_label.pack(side=TOP, fill=X)


# Function to handle window close event
def on_closing():
    root.destroy()
    sys.exit()


root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
