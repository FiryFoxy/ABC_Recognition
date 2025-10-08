# 🧠 Riconoscimento Cifre - Sito Web

Un'applicazione web moderna per il riconoscimento di cifre disegnate a mano utilizzando una rete neurale.

## 🌟 Caratteristiche

### 🎨 **Area di Disegno**
- **Canvas HTML5** per disegnare cifre con il mouse
- **Modalità Gomma** attivabile con click destro o tasto 'E'
- **Shortcut da tastiera** per controllo rapido
- **Calcolo automatico** delle percentuali per quadrante

### 🧠 **Visualizzazione Rete Neurale**
- **Architettura 4-3-3-4** visualizzata interattivamente
- **Layer colorati** per facile comprensione
- **Connessioni tra neuroni** mostrate graficamente
- **Informazioni dettagliate** sull'architettura

### 📱 **Design Moderno**
- **Responsive design** per tutti i dispositivi
- **Gradients e animazioni** per un'esperienza moderna
- **Interface intuitiva** con tab navigabili
- **Feedback visivo** in tempo reale

## 🚀 Come Utilizzare

### 1. **Aprire il Sito**
```bash
# Apri il file index.html nel tuo browser preferito
open index.html
```

### 2. **Disegnare una Cifra**
- Vai al tab "🎨 Disegna Cifra"
- Disegna una cifra chiara nella canvas
- Le percentuali dei quadranti si aggiornano automaticamente

### 3. **Classificare**
- Clicca "🔍 CLASSIFICA" o premi Invio
- Vedi il risultato (lettere a/b/c) e la confidenza

### 4. **Esplorare la Rete**
- Vai al tab "🧠 Rete Neurale"
- Osserva l'architettura 4-3-3-3
- Leggi le informazioni sui layer

### 5. **Creare un Dataset (Nuovo: 📦 Dataset)**
- Vai al tab "📦 Dataset" nel sito
- Seleziona le lettere da raccogliere: a, b, c
- Imposta "Immagini per carattere" (es. 5)
- Disegna la lettera indicata in "Prossima etichetta" e clicca "Salva campione"
- Le immagini sono ridimensionate a 28×28 in scala di grigi e salvate in memoria
- Quando hai finito, clicca "Esporta CSV" per scaricare il dataset: prima colonna `label`, seguite da 784 colonne `p0..p783`
  - Nota: ora il dataset salva solo i conteggi percentuali: `label,x1,x2,x3,x4`

## ⚙️ Modello TensorFlow.js (client-side)

- Il sito carica automaticamente TensorFlow.js dal CDN e prova a caricare un modello da `letter_recognition_model/model.json`.
- Per pubblicare su GitHub Pages, includi la cartella `letter_recognition_model` (con `model.json` e pesi binari) nella root del progetto.
- Conversione da Keras a TF.js:

```bash
pip install tensorflowjs
tensorflowjs_converter \
  --input_format keras \
  letter_recognition_model.keras \
  letter_recognition_model
```

- Una volta pubblicato, la classificazione userà il modello. Se il modello non è presente, il sito usa un fallback euristico.

## 🔢 Normalizzazione (scaler)

- Il sito prova a caricare `scaler.json` dalla root del progetto.
- Formati supportati:
  - StandardScaler: `{ "mean": [m1,m2,m3,m4], "scale": [s1,s2,s3,s4] }` con formula `(x-mean)/scale`
  - MinMax: `{ "min": [a1..a4], "scale": [k1..k4] }` con formula `(x-min)*scale`
- Se `scaler.json` non è presente, usa il fallback `x/100`.

Esempio di esportazione da Python (da `scaler.pkl` a JSON):

```python
import joblib, json
scaler = joblib.load('scaler.pkl')
cfg = {}
if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
    cfg = { 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist() }
elif hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
    cfg = { 'min': scaler.min_.tolist(), 'scale': scaler.scale_.tolist() }
with open('scaler.json', 'w') as f:
    json.dump(cfg, f)
```

## ⌨️ **Shortcut da Tastiera**

| Tasto | Azione |
|-------|--------|
| `E` o `D` | Toggle modalità gomma |
| `C` | Cancella tutto |
| `Invio` | Classifica cifra |
| `Click destro` | Attiva/disattiva gomma |

## 🏗️ **Architettura della Rete**

```
INPUT LAYER (4 neuroni)
    ↓
HIDDEN LAYER 1 (3 neuroni)
    ↓
HIDDEN LAYER 2 (3 neuroni)
    ↓
OUTPUT LAYER (3 neuroni: a–c)
```

### **Input**: x1, x2, x3, x4 (percentuali pixel neri per quadrante)
### **Output**: Probabilità per lettere a–c

## 🎯 **Algoritmo di Classificazione**

Il sito utilizza una classificazione basata su regole per dimostrare il funzionamento (a/b/c):

1. **Analisi quadranti**: Calcola percentuali di pixel neri
2. **Pattern matching**: Confronta con pattern noti
3. **Scoring**: Assegna confidenza basata su similarità
4. **Risultato**: Restituisce cifra più probabile

## 🛠️ **Tecnologie Utilizzate**

- **HTML5** - Struttura e Canvas
- **CSS3** - Styling moderno e responsive
- **JavaScript** - Logica di disegno e classificazione
- **Canvas API** - Manipolazione pixel e disegno

## 📁 **Struttura File**

```
GUI/
├── index.html          # Sito web principale
├── main.py            # Versione Python originale
├── dataset.csv        # Dataset di training (opzionale; puoi esportarne uno nuovo dal tab Dataset)
├── handwriting_ann_model.h5  # Modello Keras
├── scaler.pkl         # Scaler per normalizzazione
└── README.md          # Questa documentazione
```

## 🔧 **Personalizzazione**

### **Modificare Colori**
Edita le variabili CSS nel file `index.html`:
```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --success-color: #4CAF50;
}
```

### **Aggiungere Nuove Cifre**
Modifica la funzione `classifyDigit()` in JavaScript per supportare più cifre.

### **Cambiare Dimensioni Canvas**
```javascript
const CANVAS_SIZE = 400; // Modifica questo valore
```

## 🌐 **Compatibilità Browser**

- ✅ Chrome 60+
- ✅ Firefox 55+
- ✅ Safari 12+
- ✅ Edge 79+

## 📱 **Responsive Design**

Il sito si adatta automaticamente a:
- 📱 **Mobile** (320px+)
- 📱 **Tablet** (768px+)
- 💻 **Desktop** (1024px+)

## 🎨 **Esempi di Utilizzo**

### **Disegnare il numero 1**
- Concentra il disegno nei quadranti sinistri (x1, x3)
- Mantieni i quadranti destri (x2, x4) relativamente vuoti

### **Disegnare il numero 2**
- Usa principalmente i quadranti destri (x2, x4)
- Mantieni i quadranti sinistri (x1, x3) vuoti

### **Disegnare il numero 3**
- Distribuisci il disegno nei quadranti superiori (x1, x2)
- Mantieni i quadranti inferiori (x3, x4) vuoti

### **Disegnare il numero 4**
- Riempi tutti i quadranti in modo uniforme
- Assicurati che tutti i valori x1-x4 siano > 15%

## 🚀 **Sviluppi Futuri**

- [ ] Integrazione con modello TensorFlow.js
- [ ] Supporto per più cifre (0-9)
- [ ] Salvataggio e caricamento disegni
- [ ] Statistiche di accuratezza
- [ ] Modalità training interattiva

## 📞 **Supporto**

Per domande o problemi:
1. Controlla la console del browser per errori
2. Verifica che JavaScript sia abilitato
3. Assicurati di usare un browser moderno

---

**Creato con ❤️ per l'apprendimento del machine learning**
