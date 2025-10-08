import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib
import os
import warnings
import threading
import time

# Optimize TensorFlow loading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# --- Configurazione dei percorsi locali ---
# AGGIUSTA QUESTI PERCORSI in base a dove hai salvato i file sul tuo Mac!
MODEL_PATH = 'handwriting_ann_model.h5' 
SCALER_PATH = 'scaler.pkl' 

# --- Dimensioni Fisse ---
CANVAS_SIZE = 200 # Il canvas è 200x200 pixel
QUADRANT_SIZE = CANVAS_SIZE // 2 # Ogni quadrante è 100x100 pixel
MAX_PIXELS_PER_QUADRANT = QUADRANT_SIZE * QUADRANT_SIZE # 10000

# Sopprimi il warning sul formato .h5 (potrebbe apparire al caricamento)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Funzioni di Caricamento e Classificazione ---

def load_resources():
    """Carica il modello Keras e lo scaler joblib all'avvio dell'app."""
    try:
        print("Caricamento TensorFlow...")
        from tensorflow.keras.models import load_model
        print("TensorFlow caricato. Caricamento modello...")
        model = load_model(MODEL_PATH)
        print("Modello caricato. Caricamento scaler...")
        scaler = joblib.load(SCALER_PATH)
        print("Modello e Scaler caricati con successo.")
        return model, scaler
    except Exception as e:
        messagebox.showerror(
            "Errore di Caricamento", 
            f"Impossibile caricare il modello o lo scaler:\n{e}\n"
            f"Assicurati che i file '{MODEL_PATH}' e '{SCALER_PATH}' siano presenti nella cartella corrente."
        )
        return None, None

def classify_input(model, scaler, input_values):
    """Esegue la normalizzazione e la previsione sul nuovo input."""
    data = np.array(input_values).reshape(1, -1) # Assicura la forma (1, 4)
    input_scaled = scaler.transform(data)
    prediction_probabilities = model.predict(input_scaled, verbose=0)[0]
    predicted_index = np.argmax(prediction_probabilities)
    predicted_label = predicted_index + 1 
    confidence = prediction_probabilities[predicted_index]
    return predicted_label, confidence

# --- Interfaccia Utente (Tkinter) ---

class HandwritingClassifierApp:
    def __init__(self, master, model, scaler):
        self.master = master
        self.model = model
        self.scaler = scaler
        master.title("Riconoscimento Cifre Disegnate (Logica Dataset)")
        
        self.fixed_width = 900
        self.fixed_height = 700
        master.geometry(f"{self.fixed_width}x{self.fixed_height}")
        master.resizable(False, False) 

        # Variabili per il disegno: 0 = bianco, 1 = disegnato (nero logico)
        self.canvas_pixels = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int) 
        self.last_x, self.last_y = None, None
        self.brush_size = 8  # Increased brush size for better drawing
        self.eraser_mode = False  # New: eraser mode toggle
        self.current_x_values = [0, 0, 0, 0] # I valori x1, x2, x3, x4 scalati a 0-100

        self.create_widgets()
        self.clear_canvas() 

    def create_widgets(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.create_drawing_tab()
        self.create_neural_network_tab()

    def create_drawing_tab(self):
        """Create the drawing tab with canvas and controls"""
        # Create drawing tab
        drawing_tab = ttk.Frame(self.notebook)
        self.notebook.add(drawing_tab, text="🎨 Disegna Cifra")
        
        # Create frame for drawing tab content
        drawing_frame = ttk.Frame(drawing_tab, padding="10")
        drawing_frame.pack(fill='both', expand=True)
        
        # Create canvas and control frames
        canvas_frame = ttk.Frame(drawing_frame, width=CANVAS_SIZE, height=CANVAS_SIZE)
        canvas_frame.grid(row=0, column=0, padx=10, pady=10, sticky='n')

        control_frame = ttk.Frame(drawing_frame, padding="10")
        control_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')
        
        drawing_frame.grid_columnconfigure(0, weight=0)
        drawing_frame.grid_columnconfigure(1, weight=1)
        drawing_frame.grid_rowconfigure(0, weight=1)

        # --- Canvas di Disegno 200x200px ---
        self.canvas = tk.Canvas(canvas_frame, bg="white", width=CANVAS_SIZE, height=CANVAS_SIZE, bd=2, relief="solid")
        self.canvas.pack() 
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        self.canvas.bind("<Button-3>", self.toggle_eraser)  # Right-click to toggle eraser
        
        # Keyboard shortcuts
        self.master.bind("<KeyPress-e>", lambda e: self.toggle_eraser_mode())
        self.master.bind("<KeyPress-d>", lambda e: self.toggle_eraser_mode())
        self.master.bind("<KeyPress-c>", lambda e: self.clear_canvas_with_confirmation())
        self.master.bind("<KeyPress-Return>", lambda e: self.handle_classification())
        self.master.focus_set()  # Enable keyboard focus
        
        # Linee Guida dei Quadranti
        self.canvas.create_line(QUADRANT_SIZE, 0, QUADRANT_SIZE, CANVAS_SIZE, fill="lightgray", dash=(4, 2))
        self.canvas.create_line(0, QUADRANT_SIZE, CANVAS_SIZE, QUADRANT_SIZE, fill="lightgray", dash=(4, 2))

        # --- Sezione Controlli ---
        ttk.Label(control_frame, text="Risultato Classificazione:", font=('Helvetica', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 10))

        self.result_label = ttk.Label(control_frame, text="Cifra Riconosciuta: N/A", font=('Helvetica', 16, 'bold'), foreground='navy')
        self.result_label.grid(row=1, column=0, columnspan=2, pady=(0, 5))
        self.confidence_label = ttk.Label(control_frame, text="Confidenza: N/A", font=('Helvetica', 12))
        self.confidence_label.grid(row=2, column=0, columnspan=2, pady=(0, 20))
        
        # Visualizzazione Conteggi Quadranti (x1, x2, x3, x4)
        ttk.Label(control_frame, text="Conteggi Percentuali (0-100):", font=('Helvetica', 10, 'bold')).grid(row=3, column=0, columnspan=2, pady=(10, 5))
        
        self.x_labels = {}
        labels_text = ["x1 (TL):", "x2 (TR):", "x3 (BL):", "x4 (BR):"]
        for i, text in enumerate(labels_text):
            ttk.Label(control_frame, text=text).grid(row=4+i, column=0, sticky='w', padx=5, pady=2)
            lbl = ttk.Label(control_frame, text="0.00", font=('Monospace', 10, 'bold'))
            lbl.grid(row=4+i, column=1, sticky='ew', padx=5, pady=2)
            self.x_labels[f'x{i+1}'] = lbl

        # Instructions
        ttk.Label(control_frame, text="Istruzioni:", font=('Helvetica', 10, 'bold')).grid(row=8, column=0, columnspan=2, pady=(20, 5), sticky='w')
        ttk.Label(control_frame, text="• Disegna con il mouse", font=('Helvetica', 9)).grid(row=9, column=0, columnspan=2, pady=1, sticky='w')
        ttk.Label(control_frame, text="• Click destro = Gomma", font=('Helvetica', 9)).grid(row=10, column=0, columnspan=2, pady=1, sticky='w')
        ttk.Label(control_frame, text="• E/D = Gomma, C = Cancella, Enter = Classifica", font=('Helvetica', 9)).grid(row=11, column=0, columnspan=2, pady=1, sticky='w')
        ttk.Label(control_frame, text="• Disegna una cifra chiara", font=('Helvetica', 9)).grid(row=12, column=0, columnspan=2, pady=1, sticky='w')
        
        # Mode indicator
        self.mode_label = ttk.Label(control_frame, text="Modalità: Disegno", font=('Helvetica', 10, 'bold'), foreground='blue')
        self.mode_label.grid(row=13, column=0, columnspan=2, pady=(10, 5))

        # Pulsanti Azione
        ttk.Button(control_frame, text="CLASSIFICA", command=self.handle_classification).grid(row=14, column=0, columnspan=2, pady=(10, 5))
        ttk.Button(control_frame, text="CANCELLA TUTTO", command=self.clear_canvas_with_confirmation).grid(row=15, column=0, columnspan=2, pady=5)

    def create_neural_network_tab(self):
        """Create the neural network visualization tab"""
        # Create neural network tab
        nn_tab = ttk.Frame(self.notebook)
        self.notebook.add(nn_tab, text="🧠 Rete Neurale")
        
        # Create main frame for NN visualization
        nn_frame = ttk.Frame(nn_tab, padding="20")
        nn_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(nn_frame, text="Architettura della Rete Neurale", 
                               font=('Helvetica', 18, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Create canvas for neural network visualization
        self.nn_canvas = tk.Canvas(nn_frame, bg="white", width=800, height=500, bd=2, relief="solid")
        self.nn_canvas.pack(fill='both', expand=True, pady=(0, 20))
        
        # Bind resize event to redraw network
        self.nn_canvas.bind("<Configure>", lambda e: self.draw_neural_network())
        
        # Draw neural network
        self.draw_neural_network()
        
        # Information panel
        info_frame = ttk.LabelFrame(nn_frame, text="Informazioni sulla Rete", padding="10")
        info_frame.pack(fill='x')
        
        info_text = """
• INPUT LAYER: 4 neuroni (x1, x2, x3, x4) - Percentuali di pixel neri per quadrante
• HIDDEN LAYER 1: 3 neuroni - Prima elaborazione delle caratteristiche
• HIDDEN LAYER 2: 3 neuroni - Seconda elaborazione delle caratteristiche  
• OUTPUT LAYER: 4 neuroni (cifre 1-4) - Probabilità per ogni cifra
• ARCHITETTURA: 4-3-3-4 (Input-Hidden1-Hidden2-Output)
• ATTIVAZIONE: Funzioni di attivazione per la non-linearità
• ADDESTRAMENTO: Rete allenata su dataset di cifre scritte a mano
        """
        
        info_label = ttk.Label(info_frame, text=info_text, font=('Helvetica', 10), 
                              justify='left')
        info_label.pack(anchor='w')

    def draw_neural_network(self):
        """Draw the neural network architecture"""
        canvas = self.nn_canvas
        canvas.delete("all")  # Clear canvas
        
        # Get canvas dimensions
        width = canvas.winfo_width() or 800
        height = canvas.winfo_height() or 500
        
        # Define layer positions and sizes (dynamically calculated)
        margin = 80
        available_width = width - 2 * margin
        layer_spacing = available_width // 4
        layer_x = [margin + i * layer_spacing for i in range(4)]
        
        layer_names = ["INPUT\n(x1,x2,x3,x4)", "HIDDEN\nLAYER 1", "HIDDEN\nLAYER 2", "OUTPUT\n(1-4)"]
        layer_sizes = [4, 3, 3, 4]  # Number of neurons in each layer (4-3-3-4)
        layer_colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]  # Green, Blue, Orange, Red
        
        # Adjust neuron size based on available space
        neuron_radius = min(18, layer_spacing // 8)
        
        # Draw layers
        for layer_idx, (x, name, size, color) in enumerate(zip(layer_x, layer_names, layer_sizes, layer_colors)):
            # Draw layer label
            canvas.create_text(x, 50, text=name, font=('Helvetica', 12, 'bold'), fill='black')
            
            # Calculate neuron positions in this layer
            start_y = (height - 100) // 2 - (size * (neuron_radius * 2 + 10)) // 2 + 100
            
            for neuron_idx in range(size):
                y = start_y + neuron_idx * (neuron_radius * 2 + 10)
                
                # Draw neuron (circle)
                canvas.create_oval(x - neuron_radius, y - neuron_radius, 
                                 x + neuron_radius, y + neuron_radius,
                                 fill=color, outline='black', width=2)
                
                # Add neuron number for input layer
                if layer_idx == 0:
                    canvas.create_text(x, y, text=f"x{neuron_idx+1}", 
                                     font=('Helvetica', 10, 'bold'), fill='white')
                elif layer_idx == len(layer_x) - 1:  # Output layer
                    canvas.create_text(x, y, text=str(neuron_idx+1), 
                                     font=('Helvetica', 10, 'bold'), fill='white')
                else:
                    # Hidden layers - show neuron number
                    canvas.create_text(x, y, text=str(neuron_idx+1), 
                                     font=('Helvetica', 8, 'bold'), fill='white')
        
        # Draw connections between layers
        for layer_idx in range(len(layer_x) - 1):
            current_x = layer_x[layer_idx]
            next_x = layer_x[layer_idx + 1]
            current_size = layer_sizes[layer_idx]
            next_size = layer_sizes[layer_idx + 1]
            
            # Calculate start positions for current layer
            current_start_y = (height - 100) // 2 - (current_size * (neuron_radius * 2 + 10)) // 2 + 100
            next_start_y = (height - 100) // 2 - (next_size * (neuron_radius * 2 + 10)) // 2 + 100
            
            # Draw connections
            for i in range(current_size):
                current_y = current_start_y + i * (neuron_radius * 2 + 10)
                for j in range(next_size):
                    next_y = next_start_y + j * (neuron_radius * 2 + 10)
                    
                    # Draw connection line
                    canvas.create_line(current_x + neuron_radius, current_y,
                                     next_x - neuron_radius, next_y,
                                     fill='gray', width=1)
        
        # Add title
        canvas.create_text(width // 2, 30, text="Architettura Rete Neurale per Riconoscimento Cifre", 
                         font=('Helvetica', 16, 'bold'), fill='darkblue')
        
        # Add data flow arrows
        for i in range(len(layer_x) - 1):
            x = (layer_x[i] + layer_x[i+1]) // 2
            y = height - 40
            canvas.create_polygon(x-8, y-5, x+8, y-5, x+4, y-12, x-4, y-12, 
                                fill='darkgreen', outline='black')
            canvas.create_text(x, y-20, text="→", font=('Helvetica', 12, 'bold'), fill='darkgreen')

    # --- Metodi di Disegno (Invariati) ---
    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y
        self.draw_on_canvas(event.x, event.y)

    def draw_line(self, event):
        if self.last_x and self.last_y:
            color = "white" if self.eraser_mode else "black"
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, 
                                    width=self.brush_size, fill=color, capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw_on_canvas(event.x, event.y)
            self.last_x, self.last_y = event.x, event.y

    def stop_draw(self, event):
        self.last_x, self.last_y = None, None
        self.update_pixel_counts() # Aggiorna e calcola la percentuale

    def draw_on_canvas(self, x, y):
        # L'array canvas_pixels è 0=bianco, 1=nero. Questo simula la binarizzazione.
        for i in range(max(0, x - self.brush_size // 2), min(CANVAS_SIZE, x + self.brush_size // 2 + 1)):
            for j in range(max(0, y - self.brush_size // 2), min(CANVAS_SIZE, y + self.brush_size // 2 + 1)):
                if (i - x)**2 + (j - y)**2 <= (self.brush_size / 2)**2:
                    if 0 <= i < CANVAS_SIZE and 0 <= j < CANVAS_SIZE:
                        if self.eraser_mode:
                            self.canvas_pixels[j, i] = 0  # Erase (set to white)
                        else:
                            self.canvas_pixels[j, i] = 1  # Draw (set to black)

    def toggle_eraser(self, event):
        """Toggle between drawing and erasing mode"""
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.mode_label.config(text="Modalità: Gomma", foreground='red')
        else:
            self.mode_label.config(text="Modalità: Disegno", foreground='blue')

    def toggle_eraser_mode(self):
        """Toggle eraser mode via keyboard shortcut"""
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.mode_label.config(text="Modalità: Gomma", foreground='red')
        else:
            self.mode_label.config(text="Modalità: Disegno", foreground='blue')

    def clear_canvas_with_confirmation(self):
        """Clear canvas with confirmation dialog"""
        if messagebox.askyesno("Conferma", "Sei sicuro di voler cancellare tutto il disegno?"):
            self.clear_canvas() 

    # --- Metodo Aggiornato con Logica Dataset ---
    def update_pixel_counts(self):
        """
        Calcola la percentuale di pixel neri in ogni quadrante (0-100) 
        rispecchiando ESATTAMENTE la logica del tuo script di pre-elaborazione.
        """
        
        # Il tuo script usava un ciclo nidificato. Lo replichiamo per chiarezza logica.
        
        input_values = []
        
        # Blocchi 100x100
        block_size = QUADRANT_SIZE 
        total_pixels = MAX_PIXELS_PER_QUADRANT # 10000

        # Il nostro array self.canvas_pixels è già l'equivalente di 'img_array' 
        # dopo la binarizzazione, dove i pixel neri sono rappresentati da '1'.
        
        for i in range(2):  # righe (0, 1)
            for j in range(2):  # colonne (0, 1)
                
                # Estrazione del blocco
                block = self.canvas_pixels[i*block_size:(i+1)*block_size,
                                           j*block_size:(j+1)*block_size]

                # Conteggio dei pixel neri (che sono i valori '1' nel nostro array)
                black_pixels = np.sum(block == 1) # Nel tuo script era block == 0, qui è block == 1

                # Percentuale neri in centesimi: (black_pixels / total_pixels) * 100
                percent_black = (black_pixels / total_pixels) * 100
                
                # Arrotondamento a due decimali (come nel tuo script)
                input_values.append(round(percent_black, 2))

        self.current_x_values = input_values
        
        # Aggiorna le etichette nella GUI
        self.x_labels['x1'].config(text=f"{input_values[0]:.2f}")
        self.x_labels['x2'].config(text=f"{input_values[1]:.2f}")
        self.x_labels['x3'].config(text=f"{input_values[2]:.2f}")
        self.x_labels['x4'].config(text=f"{input_values[3]:.2f}")


    def clear_canvas(self):
        self.canvas.delete("all")
        # Disegna nuovamente le linee guida dei quadranti
        self.canvas.create_line(QUADRANT_SIZE, 0, QUADRANT_SIZE, CANVAS_SIZE, fill="lightgray", dash=(4, 2))
        self.canvas.create_line(0, QUADRANT_SIZE, CANVAS_SIZE, QUADRANT_SIZE, fill="lightgray", dash=(4, 2))
        
        self.canvas_pixels = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
        self.update_pixel_counts() 
        self.result_label.config(text="Cifra Riconosciuta: N/A", foreground='navy')
        self.confidence_label.config(text="Confidenza: N/A")

    def handle_classification(self):
        if self.model is None or self.scaler is None:
            self.result_label.config(text="Errore: Modello non caricato.", foreground='red')
            return
        
        # Calcola i valori x1-x4 e li memorizza in self.current_x_values
        self.update_pixel_counts() 
        
        predicted_label, confidence = classify_input(self.model, self.scaler, self.current_x_values)
        
        if predicted_label is not None:
            self.result_label.config(text=f"Cifra Riconosciuta: {predicted_label}", foreground='green')
            self.confidence_label.config(text=f"Confidenza: {confidence*100:.2f}%")
        else:
            self.result_label.config(text="Errore nella classificazione", foreground='red')

# --- Loading Screen ---

class LoadingScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Caricamento...")
        self.root.geometry("400x200")
        self.root.resizable(False, False)
        
        # Center the window
        self.root.eval('tk::PlaceWindow . center')
        
        # Loading message
        ttk.Label(self.root, text="Caricamento Riconoscimento Cifre", 
                 font=('Helvetica', 16, 'bold')).pack(pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=10, padx=50, fill='x')
        self.progress.start()
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Caricamento TensorFlow...", 
                                     font=('Helvetica', 10))
        self.status_label.pack(pady=10)
        
        # Loading info
        ttk.Label(self.root, text="Questo potrebbe richiedere alcuni secondi...", 
                 font=('Helvetica', 9), foreground='gray').pack(pady=5)
        
        self.model = None
        self.scaler = None
        
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update()
        
    def load_in_background(self):
        """Load resources in background thread"""
        def load():
            try:
                self.update_status("Caricamento TensorFlow...")
                from tensorflow.keras.models import load_model
                
                self.update_status("TensorFlow caricato. Caricamento modello...")
                self.model = load_model(MODEL_PATH)
                
                self.update_status("Modello caricato. Caricamento scaler...")
                self.scaler = joblib.load(SCALER_PATH)
                
                self.update_status("Caricamento completato!")
                time.sleep(1)  # Brief pause to show completion
                
                # Close loading screen and start main app
                self.root.after(0, self.start_main_app)
                
            except Exception as e:
                self.root.after(0, lambda: self.show_error(str(e)))
        
        # Start loading in background thread
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
        
    def show_error(self, error_message):
        self.root.destroy()
        messagebox.showerror(
            "Errore di Caricamento", 
            f"Impossibile caricare il modello o lo scaler:\n{error_message}\n"
            f"Assicurati che i file '{MODEL_PATH}' e '{SCALER_PATH}' siano presenti nella cartella corrente."
        )
        
    def start_main_app(self):
        self.root.destroy()
        if self.model and self.scaler:
            root = tk.Tk()
            app = HandwritingClassifierApp(root, self.model, self.scaler)
            root.mainloop()

# --- Esecuzione dell'Applicazione ---

if __name__ == "__main__":
    # Show loading screen
    loading = LoadingScreen()
    loading.load_in_background()
    loading.root.mainloop()