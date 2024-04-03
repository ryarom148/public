# Configure alternating row colors
        self.summary_tree.tag_configure("oddrow", background="lightgray")
        self.summary_tree.tag_configure("evenrow", background="white")

        # Bind the click event to the treeview
        self.summary_tree.bind("<ButtonRelease-1>", self.on_summary_row_click)

        # Bind the tab change event to update the detailed view
        self.tab_view.bind("<<NotebookTabChanged>>", self.on_tab_change)

def update_summary_table(self):
        # Clear existing table
        self.summary_tree.delete(*self.summary_tree.get_children())

        # Configure columns
        headers = ["User", "Low Risk", "Medium Risk", "High Risk"]
        self.summary_tree.configure(columns=headers, show="headings")

        # Set column headings
        for col in headers:
            self.summary_tree.heading(col, text=col)

        # Set column properties
        for col in headers:
            self.summary_tree.column(col, anchor="center", stretch=True)

        # Populate table with data from global_df
        for index, record in enumerate(global_df.itertuples()):
            values = [getattr(record, col) for col in headers]
            if index % 2 == 0:
                tag = "evenrow"
            else:
                tag = "oddrow"
            item = self.summary_tree.insert("", "end", values=values, tags=(tag,))

            if record.New:
                self.summary_tree.item(item, tags=("new", tag))
                self.flash_new_row(item)

    def flash_new_row(self, item):
        self.summary_tree.tag_configure("new", background="lightgreen")
        self.summary_tree.after(1000, lambda: self.summary_tree.tag_configure("new", background=""))
        self.summary_tree.after(2000, lambda: self.summary_tree.tag_configure("new", background="lightgreen"))
        self.summary_tree.after(3000, lambda: self.summary_tree.tag_configure("new", background=""))
def on_summary_row_click(self, event):
        # Get the selected item from the treeview
        item = self.summary_tree.focus()
        if item:
            # Get the values of the selected row
            values = self.summary_tree.item(item, "values")
            client = values[0]  # Assuming the client name is in the first column

            # Switch to Tab 2
            self.tab_view.set("Detailed Analysis")

            # Update the client dropdown and select the clicked client
            self.client_var.set(client)
            self.update_thread_dropdown()

    def on_tab_change(self, event):
        # Get the selected tab
        selected_tab = self.tab_view.tab(self.tab_view.select(), "text")
        if selected_tab == "Detailed Analysis":
            self.update_detailed_view()

    def update_detailed_view(self):
        selected_flags = [flag for flag, var in self.risk_flags.items() if var.get()]
        filtered_df = global_detail_df[global_detail_df["Thread Level"].isin(selected_flags)]

        clients = filtered_df["client"].unique().tolist()
        self.client_dropdown.configure(values=clients)

        if clients:
            if self.client_var.get() not in clients:
                self.client_var.set(clients[0])
            self.update_thread_dropdown()

