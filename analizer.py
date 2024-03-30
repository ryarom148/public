import streamlit as st
import os
import pandas as pd
import threading
import json
import logging
from multiprocessing import Pool
is_analyzing = False
# Setup Basic Logging
logging.basicConfig(filename="pst_analyzer.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Placeholder Functions for PST Analysis (replace with your actual logic)
def process_pst_file(pst_file, output_folder):
    try:
        # ... Extract email threads, save them as text files in the output_folder...
    except Exception as e:
        logging.error(f"Error processing PST file: {pst_file}", exc_info=True)

def scan_conversations(folder_path):
    try:
        # ... Analyze conversations in the folder, generate JSON ... 
        return {"user": "...", "conversations": [...]} # Example structure
    except Exception as e:
        logging.error(f"Error during security scan: {folder_path}", exc_info=True)

def analyze_pst_file(pst_file, output_folder):
    try:
        process_pst_file(pst_file, output_folder)
        analysis_result = scan_conversations(output_folder)
        return analysis_result
    except Exception as e:
        logging.error(f"Error analyzing PST file: {pst_file}", exc_info=True)  

# Global Data Structures with Locks
global_summary_table = pd.DataFrame(columns=["User", "Low Risk", "Medium Risk", "High Risk"])
global_detailed_table = [] 
summary_lock = threading.Lock()
detailed_lock = threading.Lock()

# Update Tables
def update_global_summary(summary_json):
    # ... Parse JSON and create a row for the global summary table ...
    with summary_lock:
        global global_summary_table
        global_summary_table = global_summary_table.append(summary_json, ignore_index=True) 
        summary_table_placeholder.dataframe(global_summary_table.style.highlight_max(color='yellow', axis=0))
            # Update Progress Bar: Now updates as PST files are processed
        current_count = global_summary_table.shape[0]  # Total rows in table
        total_files = st.session_state['total_pst_files']  # From state (see below)
        if total_files > 0:  # Avoid division by zero
            progress_bar.progress(current_count / total_files)

def update_global_detail(summary_json):
    with detailed_lock:
        global global_detailed_table
        global_detailed_table.append(summary_json)

# Analyze Folder
def analyze_pst_folder(folder_path, output_folder, progress_bar):
    global is_analyzing
    pst_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pst")]

    with Pool() as pool:
        pool.starmap(analyze_pst_file, zip(pst_files, [output_folder] * len(pst_files)))

    progress_bar.progress(1)

def load_global_summary():
    # ... Loading logic from CSV
    pass
def validate_folder_access(folder_path):
    """Checks if the user has read access to the specified folder.
    Feel free to customize this logic based on your security requirements.
    """
    if not os.access(folder_path, os.R_OK):
        raise PermissionError(f"User does not have read access to folder: {folder_path}")

# Streamlit UI (with Error Handling)
st.title("PST Security Analyzer")
tab1, tab2 = st.tabs(["Processing & Summary", "Detailed Analysis"])

with tab1: 
    folder_path = st.file_uploader("Select PST Folder")
    output_folder = "pst_output"  
    analysis_button = st.button("Start Analysis", disabled=is_analyzing)
    if folder_path:
        try:
            load_global_summary()
            validate_folder_access(folder_path)
            st.info(f"{len([f for f in os.listdir(folder_path) if f.endswith('.pst')])} PST files found.") 
            pst_files = [f for f in os.listdir(folder_path) if f.endswith(".pst")]
            st.session_state['total_pst_files'] = len(pst_files) # Store count in session


            if analysis_button.clicked:  # If the button is clicked
                with st.spinner("Validating access and analyzing..."):  
                    st.subheader("Progress")
                    progress_bar = st.progress(0)
                    analyze_pst_folder(folder_path, output_folder) 
                st.success("Analysis Complete!")

            st.subheader("Summary Table") 
            summary_table_placeholder = st.empty() 
        except FileNotFoundError as e:
            st.error("Error: Folder not found.")
            logging.error("Folder not found", exc_info=True)
        except Exception as e:  # Catch other potential errors
            st.error("An error occurred during analysis.")
            logging.error("Analysis error", exc_info=True)
# Tab 2 (Filtering, Dropdowns, Accordion)

st.subheader("Filtering")

risk_flags = {}
risk_flags['Low'] = st.checkbox("Low", value=True)
risk_flags['Medium'] = st.checkbox("Medium", value=True)
risk_flags['High'] = st.checkbox("High", value=True)

selected_user = None  # Variable to store the selected user

def update_user_list():
    filtered_data = global_detailed_table.copy()  

    selected_risk_flags = [flag for flag, checked in risk_flags.items() if checked]
    filtered_data = filtered_data[filtered_data['Sensitivity Flag'].isin(selected_risk_flags)]

    user_list = filtered_data['owner'].unique().tolist()  
    user_dropdown.options(user_list) 

    global selected_user
    if selected_user in user_list:
        user_dropdown.value = selected_user

def update_thread_list(newly_selected_user):
    global selected_user
    selected_user = newly_selected_user  

    filtered_data = global_detailed_table[global_detailed_table['owner'] == selected_user]
    thread_list = filtered_data['filename'].tolist() 
    thread_dropdown.options(thread_list)  

def populate_accordion_content(selected_user, selected_thread):
    if not selected_user or not selected_thread:
        return 

    filtered_data = global_detailed_table[
        (global_detailed_table['owner'] == selected_user) & 
        (global_detailed_table['filename'] == selected_thread)
    ]

    sensitivity_flag = filtered_data['Sensitivity Flag'].iloc[0]
    sensitivity_flag_label.text(f"Sensitivity Flag: {sensitivity_flag}")

    sensitivity = filtered_data['Sensitivity Explanation'].iloc[0]  
    summary = filtered_data['Summary'].iloc[0]  
    sensitivity_placeholder.text(sensitivity)
    summary_placeholder.text(summary)

    thread_path = os.path.join("pst_output", selected_thread)
    with open(thread_path, 'r') as f:
        email_thread = f.read()
    email_thread_placeholder.text(email_thread)

    user_summary_data = global_detailed_table[global_detailed_table['owner'] == selected_user]
    risk_counts = user_summary_data['Sensitivity Flag'].value_counts().to_frame().reset_index()
    risk_counts.columns = ['Risk', 'Count']
    user_summary_placeholder.dataframe(risk_counts.style.highlight_max(color='yellow', axis=0))

user_dropdown = st.selectbox("Select PST User", options=[], on_change=update_thread_list)
thread_dropdown = st.selectbox("Select Thread Subject", options=[], on_change=populate_accordion_content, args=(user_dropdown.value,))  

# Accordion for detailed view
st.subheader("Detailed View (Accordion)")
accordion = st.expander("Accordion")
with accordion:
    sensitivity_flag_label = st.empty() 
    sensitivity_placeholder = st.empty()
    summary_placeholder = st.empty()
    email_thread_placeholder = st.empty()
    user_summary_placeholder = st.empty()

st.experimental_rerun()  
