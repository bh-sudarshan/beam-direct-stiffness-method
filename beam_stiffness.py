import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class BeamAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Beam Analysis using Direct Stiffness Method")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.E = tk.DoubleVar(value=200e9)
        self.num_elements = tk.IntVar(value=1)
        self.L_list = []
        self.I_list = []
        self.loads = []
        self.element_loads = []
        self.member_forces = []
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_input_tab()
        self.create_results_tab()
        self.create_visualization_tab()
        
    def create_input_tab(self):
        """Create the input tab with all input fields"""
        input_tab = ttk.Frame(self.notebook)
        self.notebook.add(input_tab, text="Input Parameters")
        
        # Material Properties Frame
        material_frame = ttk.LabelFrame(input_tab, text="Material Properties")
        material_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(material_frame, text="Young's Modulus E (Pa):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(material_frame, textvariable=self.E).grid(row=0, column=1, padx=5, pady=5)
        
        # Element Properties Frame
        element_frame = ttk.LabelFrame(input_tab, text="Element Properties")
        element_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(element_frame, text="Number of Elements:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(element_frame, from_=1, to=20, textvariable=self.num_elements, 
                   command=self.update_element_fields).grid(row=0, column=1, padx=5, pady=5)
        
        # Frame for element length and inertia inputs
        self.element_inputs_frame = ttk.Frame(element_frame)
        self.element_inputs_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        self.create_element_input_fields()
        
        # Nodal Loads Frame
        load_frame = ttk.LabelFrame(input_tab, text="Nodal Loads")
        load_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.load_tree = ttk.Treeview(load_frame, columns=('Node', 'Load (N)'), show='headings', height=4)
        self.load_tree.heading('Node', text='Node')
        self.load_tree.heading('Load (N)', text='Load (N)')
        self.load_tree.pack(fill=tk.X, padx=5, pady=5)
        
        load_btn_frame = ttk.Frame(load_frame)
        load_btn_frame.pack(fill=tk.X)
        
        ttk.Button(load_btn_frame, text="Add Load", command=self.add_load).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_btn_frame, text="Remove Load", command=self.remove_load).pack(side=tk.LEFT, padx=5)
        
        # Element Loads Frame
        element_load_frame = ttk.LabelFrame(input_tab, text="Element Loads")
        element_load_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.element_load_tree = ttk.Treeview(element_load_frame, 
                                            columns=('Element', 'Type', 'Magnitude', 'Position'), 
                                            show='headings', height=4)
        self.element_load_tree.heading('Element', text='Element')
        self.element_load_tree.heading('Type', text='Type')
        self.element_load_tree.heading('Magnitude', text='Magnitude')
        self.element_load_tree.heading('Position', text='Position (m)')
        self.element_load_tree.pack(fill=tk.X, padx=5, pady=5)
        
        element_load_btn_frame = ttk.Frame(element_load_frame)
        element_load_btn_frame.pack(fill=tk.X)
        
        ttk.Button(element_load_btn_frame, text="Add Element Load", command=self.add_element_load).pack(side=tk.LEFT, padx=5)
        ttk.Button(element_load_btn_frame, text="Remove Element Load", command=self.remove_element_load).pack(side=tk.LEFT, padx=5)
        
        # Matrix Reduction Frame
        reduction_frame = ttk.LabelFrame(input_tab, text="Matrix Reduction")
        reduction_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(reduction_frame, text="DOFs to Remove (comma-separated, 1-based indexing):").pack(pady=5)
        
        self.dofs_to_remove_entry = ttk.Entry(reduction_frame)
        self.dofs_to_remove_entry.pack(fill=tk.X, padx=5, pady=5)
        
        # Analysis Button
        ttk.Button(input_tab, text="Run Analysis", command=self.run_analysis).pack(pady=10)
    
    def create_element_input_fields(self):
        """Create input fields for element lengths and inertias"""
        # Clear any existing widgets
        for widget in self.element_inputs_frame.winfo_children():
            widget.destroy()
            
        num_elements = self.num_elements.get()
        self.L_list = [tk.DoubleVar(value=1.0) for _ in range(num_elements)]
        self.I_list = [tk.DoubleVar(value=1e-6) for _ in range(num_elements)]
        
        ttk.Label(self.element_inputs_frame, text="Element").grid(row=0, column=0, padx=5)
        ttk.Label(self.element_inputs_frame, text="Length (m)").grid(row=0, column=1, padx=5)
        ttk.Label(self.element_inputs_frame, text="Moment of Inertia (m⁴)").grid(row=0, column=2, padx=5)
        
        for i in range(num_elements):
            ttk.Label(self.element_inputs_frame, text=f"Element {i+1}").grid(row=i+1, column=0, padx=5, pady=2)
            ttk.Entry(self.element_inputs_frame, textvariable=self.L_list[i], width=10).grid(row=i+1, column=1, padx=5, pady=2)
            ttk.Entry(self.element_inputs_frame, textvariable=self.I_list[i], width=15).grid(row=i+1, column=2, padx=5, pady=2)
    
    def update_element_fields(self):
        """Update the element input fields when number of elements changes"""
        self.create_element_input_fields()
        
    def add_load(self):
        """Add a nodal load to the treeview"""
        num_nodes = self.num_elements.get() + 1
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Nodal Load")
        
        ttk.Label(dialog, text="Node:").grid(row=0, column=0, padx=5, pady=5)
        node_spin = ttk.Spinbox(dialog, from_=0, to=num_nodes-1)
        node_spin.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Load (N):").grid(row=1, column=0, padx=5, pady=5)
        load_entry = ttk.Entry(dialog)
        load_entry.grid(row=1, column=1, padx=5, pady=5)
        
        def save_load():
            try:
                node = int(node_spin.get())
                load = float(load_entry.get())
                self.load_tree.insert('', tk.END, values=(node, load))
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        ttk.Button(dialog, text="Add", command=save_load).grid(row=2, column=0, columnspan=2, pady=5)
    
    def remove_load(self):
        """Remove selected nodal load"""
        selected = self.load_tree.selection()
        if selected:
            self.load_tree.delete(selected)
    
    def add_element_load(self):
        """Add an element load to the treeview"""
        num_elements = self.num_elements.get()
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Element Load")
        
        ttk.Label(dialog, text="Element:").grid(row=0, column=0, padx=5, pady=5)
        element_spin = ttk.Spinbox(dialog, from_=1, to=num_elements)
        element_spin.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Type:").grid(row=1, column=0, padx=5, pady=5)
        type_var = tk.StringVar()
        ttk.Combobox(dialog, textvariable=type_var, values=["udl", "point"]).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Magnitude:").grid(row=2, column=0, padx=5, pady=5)
        mag_entry = ttk.Entry(dialog)
        mag_entry.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Position (m):").grid(row=3, column=0, padx=5, pady=5)
        pos_entry = ttk.Entry(dialog)
        pos_entry.grid(row=3, column=1, padx=5, pady=5)
        
        def save_element_load():
            try:
                element = int(element_spin.get())
                load_type = type_var.get()
                magnitude = float(mag_entry.get())
                position = pos_entry.get() if load_type == "point" else "N/A"
                
                if load_type == "point":
                    position = float(position)
                    if position > self.L_list[element-1].get():
                        messagebox.showerror("Error", "Position cannot be greater than element length")
                        return
                
                self.element_load_tree.insert('', tk.END, 
                                            values=(element, load_type, magnitude, position))
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        ttk.Button(dialog, text="Add", command=save_element_load).grid(row=4, column=0, columnspan=2, pady=5)
    
    def remove_element_load(self):
        """Remove selected element load"""
        selected = self.element_load_tree.selection()
        if selected:
            self.element_load_tree.delete(selected)
    
    def create_results_tab(self):
        """Create the results tab to display analysis results"""
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results")
        
        # Notebook for different result types
        self.results_notebook = ttk.Notebook(self.results_tab)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Element Stiffness Matrices Tab
        self.element_matrices_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.element_matrices_tab, text="Element Matrices")
        
        self.element_matrices_text = tk.Text(self.element_matrices_tab, wrap=tk.NONE)
        scroll_y = ttk.Scrollbar(self.element_matrices_tab, orient=tk.VERTICAL, command=self.element_matrices_text.yview)
        scroll_x = ttk.Scrollbar(self.element_matrices_tab, orient=tk.HORIZONTAL, command=self.element_matrices_text.xview)
        self.element_matrices_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.element_matrices_text.pack(fill=tk.BOTH, expand=True)
        
        # Global Stiffness Matrix Tab
        self.global_matrix_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.global_matrix_tab, text="Global Matrix")
        
        self.global_matrix_text = tk.Text(self.global_matrix_tab, wrap=tk.NONE)
        scroll_y = ttk.Scrollbar(self.global_matrix_tab, orient=tk.VERTICAL, command=self.global_matrix_text.yview)
        scroll_x = ttk.Scrollbar(self.global_matrix_tab, orient=tk.HORIZONTAL, command=self.global_matrix_text.xview)
        self.global_matrix_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.global_matrix_text.pack(fill=tk.BOTH, expand=True)
        
        # Reduced System Tab
        self.reduced_system_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.reduced_system_tab, text="Reduced System")
        
        self.reduced_system_text = tk.Text(self.reduced_system_tab, wrap=tk.NONE)
        scroll_y = ttk.Scrollbar(self.reduced_system_tab, orient=tk.VERTICAL, command=self.reduced_system_text.yview)
        scroll_x = ttk.Scrollbar(self.reduced_system_tab, orient=tk.HORIZONTAL, command=self.reduced_system_text.xview)
        self.reduced_system_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.reduced_system_text.pack(fill=tk.BOTH, expand=True)
        
        # Displacements Tab
        self.displacements_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.displacements_tab, text="Displacements")
        
        self.displacements_text = tk.Text(self.displacements_tab, wrap=tk.NONE)
        scroll_y = ttk.Scrollbar(self.displacements_tab, orient=tk.VERTICAL, command=self.displacements_text.yview)
        scroll_x = ttk.Scrollbar(self.displacements_tab, orient=tk.HORIZONTAL, command=self.displacements_text.xview)
        self.displacements_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.displacements_text.pack(fill=tk.BOTH, expand=True)
        
        # Member Forces Tab
        self.member_forces_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.member_forces_tab, text="Member Forces")
        
        self.member_forces_text = tk.Text(self.member_forces_tab, wrap=tk.NONE)
        scroll_y = ttk.Scrollbar(self.member_forces_tab, orient=tk.VERTICAL, command=self.member_forces_text.yview)
        scroll_x = ttk.Scrollbar(self.member_forces_tab, orient=tk.HORIZONTAL, command=self.member_forces_text.xview)
        self.member_forces_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.member_forces_text.pack(fill=tk.BOTH, expand=True)
        
        # Save Results Button
        ttk.Button(self.results_tab, text="Save Results", command=self.save_results).pack(pady=10)
    
    def create_visualization_tab(self):
        """Create tab for visualizing the beam and results"""
        self.visualization_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_tab, text="Visualization")
        
        # Matplotlib figure for beam diagram
        self.beam_fig = Figure(figsize=(10, 4), dpi=100)
        self.beam_ax = self.beam_fig.add_subplot(111)
        self.beam_canvas = FigureCanvasTkAgg(self.beam_fig, self.visualization_tab)
        self.beam_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib figure for results
        self.results_fig = Figure(figsize=(10, 4), dpi=100)
        self.results_ax = self.results_fig.add_subplot(111)
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, self.visualization_tab)
        self.results_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Save Plot Button
        ttk.Button(self.visualization_tab, text="Save Plots", command=self.save_plots).pack(pady=10)
    
    def beam_element_stiffness(self, E, I, L):
        """Calculate the stiffness matrix for a beam element"""
        factor = E * I / L**3
        return factor * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
    
    def calculate_fixed_end_forces(self, element, load_type, magnitude, position, L):
        """Calculate fixed-end forces for an element"""
        pe_m = np.zeros((4, 1))
        
        if load_type == "udl":
            w = magnitude
            V = w * L / 2
            M = w * L**2 / 12
            pe_m[0] += V
            pe_m[1] += M
            pe_m[2] += V
            pe_m[3] -= M
        
        elif load_type == "point":
            P = magnitude
            a = position
            b = L - a

            V_A = P * b**2 * (3*a + b) / L**3
            V_B = P * a**2 * (3*b + a) / L**3
            M_A = P * a * b**2 / L**2
            M_B = -P * a**2 * b / L**2

            pe_m[0] += V_A
            pe_m[1] += M_A
            pe_m[2] += V_B
            pe_m[3] += M_B
        
        return pe_m
    
    def run_analysis(self):
        """Perform the beam analysis"""
        try:
            # Get input values
            E = self.E.get()
            num_elements = self.num_elements.get()
            L_list = [L.get() for L in self.L_list]
            I_list = [I.get() for I in self.I_list]
            
            # Check inputs
            if num_elements <= 0:
                raise ValueError("Number of elements must be positive")
            if any(L <= 0 for L in L_list):
                raise ValueError("Element lengths must be positive")
            if any(I <= 0 for I in I_list):
                raise ValueError("Moment of inertia values must be positive")
            
            num_nodes = num_elements + 1
            dof_per_node = 2
            total_dof = num_nodes * dof_per_node
            
            # Get nodal loads
            self.loads = []
            for item in self.load_tree.get_children():
                node, load = self.load_tree.item(item, 'values')
                self.loads.append((int(node), float(load)))
            
            # Get element loads
            self.element_loads = []
            for item in self.element_load_tree.get_children():
                element, load_type, magnitude, position = self.element_load_tree.item(item, 'values')
                element = int(element)
                magnitude = float(magnitude)
                if load_type == "point":
                    position = float(position)
                self.element_loads.append((element, load_type, magnitude, position))
            
            # Step 1: Compute element stiffness matrices
            element_matrices = []
            self.element_matrices_text.delete(1.0, tk.END)
            self.element_matrices_text.insert(tk.END, "=== ELEMENT STIFFNESS MATRICES ===\n\n")
            
            for i in range(num_elements):
                ke = self.beam_element_stiffness(E, I_list[i], L_list[i])
                element_matrices.append(ke)
                
                self.element_matrices_text.insert(tk.END, f"Element {i+1} (L={L_list[i]} m, I={I_list[i]} m⁴):\n")
                self.element_matrices_text.insert(tk.END, np.array2string(ke, precision=4, suppress_small=True))
                self.element_matrices_text.insert(tk.END, "\n\n")
            
            # Step 2: Assemble global stiffness matrix
            K_global = np.zeros((total_dof, total_dof))
            self.global_matrix_text.delete(1.0, tk.END)
            self.global_matrix_text.insert(tk.END, "=== GLOBAL STIFFNESS MATRIX ===\n\n")
            
            for e in range(num_elements):
                ke = element_matrices[e]
                start = e * dof_per_node
                for i in range(4):
                    for j in range(4):
                        K_global[start+i, start+j] += ke[i, j]
            
            self.global_matrix_text.insert(tk.END, np.array2string(K_global, precision=4, suppress_small=True))
            
            # Step 3: Get DOFs to remove from user input
            dofs_to_remove_str = self.dofs_to_remove_entry.get()
            if not dofs_to_remove_str.strip():
                raise ValueError("Please specify DOFs to remove for matrix reduction")
            
            try:
                dofs_to_remove = [int(dof.strip()) - 1 for dof in dofs_to_remove_str.split(',')]
            except ValueError:
                raise ValueError("Please enter valid DOF numbers (comma-separated integers)")
            
            # Validate DOF numbers
            for dof in dofs_to_remove:
                if dof < 0 or dof >= total_dof:
                    raise ValueError(f"DOF {dof+1} is out of range (1-{total_dof})")
            
            # Determine rows and columns to keep
            keep_rows = [i for i in range(K_global.shape[0]) if i not in dofs_to_remove]
            keep_cols = [i for i in range(K_global.shape[1]) if i not in dofs_to_remove]
            
            # Step 4: Create reduced matrix
            K_reduced = K_global[np.ix_(keep_rows, keep_cols)]
            
            self.reduced_system_text.delete(1.0, tk.END)
            self.reduced_system_text.insert(tk.END, "=== REDUCED STIFFNESS MATRIX ===\n\n")
            self.reduced_system_text.insert(tk.END, np.array2string(K_reduced, precision=4, suppress_small=True))
            self.reduced_system_text.insert(tk.END, "\n\n")
            
            # Step 5: Fixed-End Forces and Load Vector Calculation
            Pe = np.zeros((total_dof, 1))
            
            for element, load_type, magnitude, position in self.element_loads:
                L = L_list[element-1]
                start_dof = (element-1) * dof_per_node
                element_dofs = [start_dof, start_dof+1, start_dof+2, start_dof+3]
                
                pe_m = self.calculate_fixed_end_forces(element, load_type, magnitude, position, L)
                
                Pe[element_dofs[0]] += pe_m[0]
                Pe[element_dofs[1]] += pe_m[1]
                Pe[element_dofs[2]] += pe_m[2]
                Pe[element_dofs[3]] += pe_m[3]
            
            # External Joint Load Vector (P0)
            P0 = np.zeros((total_dof, 1))
            for node, load in self.loads:
                P0[node * dof_per_node, 0] += load
            
            # Net Load Vector
            Pr = P0 - Pe
            
            # Reduced Load Vector
            Pr_reduced = Pr[keep_rows]
            
            self.reduced_system_text.insert(tk.END, "=== REDUCED LOAD VECTOR ===\n\n")
            self.reduced_system_text.insert(tk.END, np.array2string(Pr_reduced, precision=4, suppress_small=True))
            self.reduced_system_text.insert(tk.END, "\n\n")
            
            # Step 6: Solve for displacement vector (Δr)
            try:
                delta_r = np.linalg.solve(K_reduced, Pr_reduced)
                
                self.displacements_text.delete(1.0, tk.END)
                self.displacements_text.insert(tk.END, "=== DISPLACEMENT SOLUTION ===\n\n")
                self.displacements_text.insert(tk.END, np.array2string(delta_r, precision=6, suppress_small=True))
                self.displacements_text.insert(tk.END, "\n\n")
                
                # Optional: show in terms of 1/(EI) scaling
                self.displacements_text.insert(tk.END, "=== SCALED SOLUTION (1/EI) ===\n\n")
                delta_scaled = delta_r / (E * I_list[0])
                self.displacements_text.insert(tk.END, np.array2string(delta_scaled, precision=6, suppress_small=True))
                
                # Step 7: Calculate member forces
                self.member_forces = []
                self.member_forces_text.delete(1.0, tk.END)
                self.member_forces_text.insert(tk.END, "=== MEMBER FORCES CALCULATION ===\n\n")
                
                for i in range(num_elements):
                    self.member_forces_text.insert(tk.END, f"--- Member {i+1} ---\n")
                    
                    ke = element_matrices[i]
                    L = L_list[i]
                    start_dof = i * dof_per_node
                    elem_dofs = [start_dof, start_dof+1, start_dof+2, start_dof+3]
                    
                    # Get fixed-end forces for this element
                    pe_m = np.zeros((4, 1))
                    for element, load_type, magnitude, position in self.element_loads:
                        if element == i+1:  # Element numbers are 1-based in input
                            pe_m += self.calculate_fixed_end_forces(element, load_type, magnitude, position, L)
                    
                    # Get local displacements from global Δr
                    delta_m = np.zeros((4, 1))
                    for j in range(4):
                        global_dof = elem_dofs[j]
                        if global_dof in keep_rows:
                            idx = keep_rows.index(global_dof)
                            delta_m[j] = delta_r[idx]
                        else:
                            delta_m[j] = 0
                    
                    # Compute member force vector
                    p_star = ke @ delta_m + pe_m
                    self.member_forces.append(p_star)
                    
                    self.member_forces_text.insert(tk.END, "Fixed-end force vector [Pe_m]:\n")
                    self.member_forces_text.insert(tk.END, np.array2string(pe_m, precision=2))
                    self.member_forces_text.insert(tk.END, "\nLocal displacement vector [Δm]:\n")
                    self.member_forces_text.insert(tk.END, np.array2string(delta_m, precision=6))
                    self.member_forces_text.insert(tk.END, "\nMember force vector [P*_m]:\n")
                    self.member_forces_text.insert(tk.END, np.array2string(p_star, precision=2))
                    self.member_forces_text.insert(tk.END, "\n\n")
                
                # Visualization
                self.visualize_beam(L_list)
                self.visualize_results()
                
            except np.linalg.LinAlgError:
                messagebox.showerror("Error", "Reduced stiffness matrix is singular or not invertible.")
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis:\n{str(e)}")
    
    def visualize_beam(self, L_list):
        """Visualize the beam with supports and loads"""
        self.beam_ax.clear()
        
        total_length = sum(L_list)
        node_positions = [0]
        for L in L_list:
            node_positions.append(node_positions[-1] + L)
        
        # Draw the beam
        self.beam_ax.plot([0, total_length], [0, 0], 'k-', linewidth=3)
        
        # Draw nodes
        for i, pos in enumerate(node_positions):
            self.beam_ax.plot(pos, 0, 'bo', markersize=8)
            self.beam_ax.text(pos, -0.2, f"N{i}", ha='center')
        
        # Draw nodal loads
        for node, load in self.loads:
            x = node_positions[node]
            if load < 0:  # Downward load
                self.beam_ax.arrow(x, 0.5, 0, -0.4, head_width=0.1, head_length=0.1, fc='r', ec='r')
                self.beam_ax.text(x, 0.55, f"{abs(load)} N", ha='center', color='r')
            else:  # Upward load
                self.beam_ax.arrow(x, -0.5, 0, 0.4, head_width=0.1, head_length=0.1, fc='r', ec='r')
                self.beam_ax.text(x, -0.55, f"{load} N", ha='center', color='r')
        
        # Draw element loads
        for element, load_type, magnitude, position in self.element_loads:
            x_start = node_positions[element-1]
            x_end = node_positions[element]
            L = x_end - x_start
            
            if load_type == "udl":
                # Draw UDL
                x = np.linspace(x_start, x_end, 10)
                y = np.zeros_like(x)
                self.beam_ax.plot(x, y + 0.1, 'r-')
                for xi in x[::2]:
                    self.beam_ax.arrow(xi, 0.1, 0, -0.09, head_width=0.05, head_length=0.02, fc='r', ec='r')
                self.beam_ax.text((x_start+x_end)/2, 0.15, f"UDL: {magnitude} N/m", ha='center', color='r')
            
            elif load_type == "point":
                # Draw point load
                x_load = x_start + position
                if magnitude < 0:  # Downward load
                    self.beam_ax.arrow(x_load, 0.5, 0, -0.4, head_width=0.1, head_length=0.1, fc='r', ec='r')
                    self.beam_ax.text(x_load, 0.55, f"{abs(magnitude)} N", ha='center', color='r')
                else:  # Upward load
                    self.beam_ax.arrow(x_load, -0.5, 0, 0.4, head_width=0.1, head_length=0.1, fc='r', ec='r')
                    self.beam_ax.text(x_load, -0.55, f"{magnitude} N", ha='center', color='r')
        
        self.beam_ax.set_title("Beam Diagram with Loads")
        self.beam_ax.set_xlabel("Position (m)")
        self.beam_ax.set_ylim(-1, 1)
        self.beam_ax.grid(True)
        self.beam_ax.set_aspect('equal', adjustable='box')
        self.beam_canvas.draw()
    
    def visualize_results(self):
        """Visualize the analysis results (displacements)"""
        # This is a placeholder - you would implement actual result visualization here
        self.results_ax.clear()
        self.results_ax.set_title("Analysis Results")
        self.results_ax.set_xlabel("Position (m)")
        self.results_ax.set_ylabel("Displacement (m)")
        self.results_ax.grid(True)
        self.results_canvas.draw()
    
    def save_results(self):
        """Save the analysis results to a file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    # Write element matrices
                    f.write(self.element_matrices_text.get(1.0, tk.END))
                    
                    # Write global matrix
                    f.write("\n" + "="*50 + "\n")
                    f.write(self.global_matrix_text.get(1.0, tk.END))
                    
                    # Write reduced system
                    f.write("\n" + "="*50 + "\n")
                    f.write(self.reduced_system_text.get(1.0, tk.END))
                    
                    # Write displacements
                    f.write("\n" + "="*50 + "\n")
                    f.write(self.displacements_text.get(1.0, tk.END))
                    
                    # Write member forces
                    f.write("\n" + "="*50 + "\n")
                    f.write(self.member_forces_text.get(1.0, tk.END))
                
                messagebox.showinfo("Success", "Results saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
    
    def save_plots(self):
        """Save the visualization plots to files"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                # Save beam diagram
                beam_path = file_path.replace(".png", "_beam.png")
                self.beam_fig.savefig(beam_path, dpi=300, bbox_inches='tight')
                
                # Save results plot
                results_path = file_path.replace(".png", "_results.png")
                self.results_fig.savefig(results_path, dpi=300, bbox_inches='tight')
                
                messagebox.showinfo("Success", f"Plots saved as:\n{beam_path}\n{results_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plots:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BeamAnalysisApp(root)
    root.mainloop()
