import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import multiprocessing as mp
#from shared import global_df, update_callback
def update_global_table(df):
    
    global global_df
    with lock:
        # Update the "New" column for previously new records
        global_df.loc[global_df['New'], 'New'] = False
        # The new will be identified for flashing the new row
        df["New"] = True
        # Adding a new record to the global DataFrame
        new_record = df.copy()
        global_df = pd.concat([global_df, new_record], ignore_index=True)

        # Invoke the callback function to update the table view and progress
        if update_callback:
            update_callback()
            progress = len(global_df) / total_files
            app.update_progress(progress)

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # Register the callback function
        global update_callback
        update_callback = self.update_summary_table

        self.title("Email Analysis")
        self.geometry(f"{1200}x{600}")

        # Create tabs
        self.tab_view = ctk.CTkTabview(self, width=1200, height=600)
        self.tab_view.pack(padx=20, pady=20)

        self.tab_summary = self.tab_view.add("Processing & Summary")
        self.tab_detailed = self.tab_view.add("Detailed Analysis")

        # Tab 1: Processing & Summary
        self.folder_path = ctk.StringVar()
        self.file_type = ctk.StringVar(value="pst")
        self.processing_info = ctk.StringVar()

        # Folder selection frame
        folder_frame = ctk.CTkFrame(self.tab_summary)
        folder_frame.pack(pady=10)

        ctk.CTkLabel(folder_frame, text="Select Folder:").pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(folder_frame, text="Browse", command=self.select_folder).pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(folder_frame, textvariable=self.folder_path, width=300).pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(folder_frame, text="PST", variable=self.file_type, value="pst").pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(folder_frame, text="Text", variable=self.file_type, value="text").pack(side=tk.LEFT, padx=5)

        # Progress frame
        progress_frame = ctk.CTkFrame(self.tab_summary)
        progress_frame.pack(pady=10)

        ctk.CTkLabel(progress_frame, textvariable=self.processing_info).pack(side=tk.LEFT, padx=5)
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=400)
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.process_button = ctk.CTkButton(progress_frame, text="Process", command=self.start_processing)
        self.process_button.pack(side=tk.LEFT, padx=5)

        # Summary Table
        self.summary_table = ctk.CTkScrollableFrame(self.tab_summary, width=1000, height=400)
        self.summary_table.pack(pady=10)

        # Tab 2: Detailed Analysis
        self.risk_flags = {
            "low": ctk.BooleanVar(value=False),
            "medium": ctk.BooleanVar(value=True),
            "high": ctk.BooleanVar(value=True)
        }

        # Risk flag checkboxes
        risk_flag_frame = ctk.CTkFrame(self.tab_detailed)
        risk_flag_frame.pack(pady=10)

        ctk.CTkCheckBox(risk_flag_frame, text="Low", variable=self.risk_flags["low"], command=self.update_detailed_view).pack(side=tk.LEFT, padx=5)
        ctk.CTkCheckBox(risk_flag_frame, text="Medium", variable=self.risk_flags["medium"], command=self.update_detailed_view).pack(side=tk.LEFT, padx=5)
        ctk.CTkCheckBox(risk_flag_frame, text="High", variable=self.risk_flags["high"], command=self.update_detailed_view).pack(side=tk.LEFT, padx=5)

        # Client and email thread dropdowns
        dropdown_frame = ctk.CTkFrame(self.tab_detailed)
        dropdown_frame.pack(pady=10)

        self.client_var = ctk.StringVar()
        self.thread_var = ctk.StringVar()

        self.client_dropdown = ctk.CTkOptionMenu(dropdown_frame, variable=self.client_var, command=self.update_thread_dropdown)
        self.client_dropdown.pack(side=tk.LEFT, padx=5)
        self.thread_dropdown = ctk.CTkOptionMenu(dropdown_frame, variable=self.thread_var, command=self.update_detailed_text)
        self.thread_dropdown.pack(side=tk.LEFT, padx=5)

        # Risk flag label
        self.risk_flag_label = ctk.CTkLabel(self.tab_detailed, text="")
        self.risk_flag_label.pack(pady=10)

        # Email thread text and sensitivity explanation
        text_frame = ctk.CTkFrame(self.tab_detailed)
        text_frame.pack(pady=10)

        self.thread_text = ctk.CTkTextbox(text_frame, width=600, height=400)
        self.thread_text.pack(side=tk.LEFT, padx=5)
        self.sensitivity_text = ctk.CTkTextbox(text_frame, width=400, height=400)
        self.sensitivity_text.pack(side=tk.LEFT, padx=5)

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_path.set(folder_path)
            self.check_folder_access()

    def check_folder_access(self):
        folder_path = self.folder_path.get()
        if os.access(folder_path, os.R_OK | os.W_OK):
            self.processing_info.set("Folder access granted.")
            self.process_button.configure(state=tk.NORMAL)
            self.count_files()
        else:
            self.processing_info.set("No access to the selected folder.")
            self.process_button.configure(state=tk.DISABLED)

    def count_files(self):
        folder_path = self.folder_path.get()
        file_type = self.file_type.get()

        if file_type == "pst":
            pst_files = [f for f in os.listdir(folder_path) if f.endswith(".pst")]
            self.processing_info.set(f"Number of PST files: {len(pst_files)}")
        else:
            folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            self.processing_info.set(f"Number of folders (email box owners): {len(folders)}")

    def start_processing(self):
        folder_path = self.folder_path.get()
        file_type = self.file_type.get()

        self.process_button.configure(state=tk.DISABLED)

        # Start processing in a separate thread
        processing_thread = mp.Process(target=process_files, args=(folder_path, file_type))
        processing_thread.start()

        self.monitor_processing(processing_thread)
    def start_processing(self):
        folder_path = self.folder_path.get()
        file_type = self.file_type.get()

        self.process_button.configure(state=tk.DISABLED)

        # Start processing in a separate thread
        processing_thread = mp.Process(target=process_files, args=(folder_path, file_type))
        processing_thread.start()

    def update_progress(self, progress):
        self.progress_bar.set(progress)
        self.processing_info.set(f"Processing... ({progress * 100:.2f}%)")
    # def monitor_processing(self, processing_thread):
    #     if processing_thread.is_alive():
    #         self.update_progress()
    #         self.after(100, self.monitor_processing, processing_thread)
    #     else:
    #         self.process_button.configure(state=tk.NORMAL)
    #         self.update_summary_table()
    #         self.update_detailed_view()

    def update_progress(self):
        # Update the progress bar based on the progress of file processing
        # You can calculate the progress percentage based on the number of processed files
        progress = 0.5  # Example progress value
        self.progress_bar.set(progress)
        self.processing_info.set(f"Processing... ({progress * 100:.2f}%)")

    # def update_summary_table(self):
    #     # Clear existing table
    #     for widget in self.summary_table.winfo_children():
    #         widget.destroy()

    #     # Create table headers
    #     headers = ["User", "Low Risk", "Medium Risk", "High Risk"]
    #     for col, header in enumerate(headers):
    #         header_label = ctk.CTkLabel(self.summary_table, text=header, font=("Arial", 12, "bold"))
    #         header_label.grid(row=0, column=col, padx=10, pady=5, sticky="w")

    #     # Populate table with data from global_df
    #     for row, record in global_df.iterrows():
    #         for col, value in enumerate(record):
    #             cell_label = ctk.CTkLabel(self.summary_table, text=str(value))
    #             cell_label.grid(row=row + 1, column=col, padx=10, pady=5, sticky="w")

    #             if record["New"]:
    #                 cell_label.configure(font=("Arial", 12, "bold"), text_color="green")
    def update_summary_table(self):
        # Clear existing table
        for widget in self.summary_table.winfo_children():
            widget.destroy()

        # Create table headers
        headers = ["User", "Low Risk", "Medium Risk", "High Risk"]
        for col, header in enumerate(headers):
            header_label = ctk.CTkLabel(self.summary_table, text=header, font=("Arial", 12, "bold"))
            header_label.grid(row=0, column=col, padx=10, pady=5, sticky="w")

        # Populate table with data from global_df
        for row, record in global_df.iterrows():
            for col, value in enumerate(record):
                cell_label = ctk.CTkLabel(self.summary_table, text=str(value))
                cell_label.grid(row=row + 1, column=col, padx=10, pady=5, sticky="w")

                if record["New"]:
                    cell_label.configure(font=("Arial", 12, "bold"), text_color="green")
                    self.summary_table.after(1000, lambda: cell_label.configure(text_color="black"))

    def update_detailed_view(self):
        selected_flags = [flag for flag, var in self.risk_flags.items() if var.get()]
        filtered_df = global_detail_df[global_detail_df["Thread Level"].isin(selected_flags)]

        clients = filtered_df["client"].unique().tolist()
        self.client_dropdown.configure(values=clients)

        if clients:
            self.client_var.set(clients[0])
            self.update_thread_dropdown()

    # def update_thread_dropdown(self, *args):
    #     selected_client = self.client_var.get()
    #     selected_flags = [flag for flag, var in self.risk_flags.items() if var.get()]
    #     filtered_df = global_detail_df[(global_detail_df["client"] == selected_client) &
    #                                    (global_detail_df["Thread Level"].isin(selected_flags))]

    #     threads = filtered_df["filename"].tolist()
    #     self.thread_dropdown.configure(values=threads)

    #     if threads:
    #         self.thread_var.set(threads[0])
    #         self.update_detailed_text()

    def update_thread_dropdown(self, *args):
        selected_client = self.client_var.get()
        selected_flags = [flag for flag, var in self.risk_flags.items() if var.get()]
        filtered_df = global_detail_df[(global_detail_df["client"] == selected_client) &
                                       (global_detail_df["Thread Level"].isin(selected_flags))]

        threads = [os.path.splitext(os.path.basename(filename))[0] for filename in filtered_df["filename"]]
        self.thread_dropdown.configure(values=threads)

        if threads:
            self.thread_var.set(threads[0])
            self.update_detailed_text()

    def update_detailed_text(self, *args):
        selected_thread = self.thread_var.get()
        thread_row = global_detail_df[global_detail_df["filename"].apply(lambda x: os.path.splitext(os.path.basename(x))[0] == selected_thread)].iloc[0]

        self.risk_flag_label.configure(text=f"Risk Flag: {thread_row['Thread Level']}")

        # Update email thread text
        with open(thread_row["filename"], "r") as file:
            thread_text = file.read()
        self.thread_text.delete("1.0", tk.END)
        self.thread_text.insert(tk.END, thread_text)

        # Update sensitivity explanation
        self.sensitivity_text.delete("1.0", tk.END)
        self.sensitivity_text.insert(tk.END, thread_row["Sensitivity_Expnation"])

    def update_detailed_text(self, *args):
        selected_thread = self.thread_var.get()
        thread_row = global_detail_df[global_detail_df["filename"] == selected_thread].iloc[0]

        self.risk_flag_label.configure(text=f"Risk Flag: {thread_row['Thread Level']}")

        # Update email thread text
        with open(thread_row["filename"], "r") as file:
            thread_text = file.read()
        self.thread_text.delete("1.0", tk.END)
        self.thread_text.insert(tk.END, thread_text)

        # Update sensitivity explanation
        self.sensitivity_text.delete("1.0", tk.END)
        self.sensitivity_text.insert(tk.END, thread_row["Sensitivity_Expnation"])

if __name__ == "__main__":
    app = App()
    app.mainloop()