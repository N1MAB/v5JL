# AI Jupyter Notebook v5 - JupyterLab Edition

Experimental version using JupyterLab as frontend with AI-powered backend from v4.

## What's Different from v4?

### v4 (Custom Frontend)
- Custom-built CodeMirror interface
- Custom notebook rendering
- All-in-one single page app
- ~2000 lines of custom JS

### v5JL (JupyterLab)
- Professional JupyterLab interface
- Battle-tested UI with extensions
- Reuses AI backend from v4
- Much less custom code to maintain

## Architecture

```
v5JL/
├── backend/              # Flask API (from v4)
│   └── app.py           # AI code generation, validation, execution
├── chat/                 # AI Chat Interface
│   ├── index.html       # Chat UI
│   └── server.py        # Web server for chat
├── notebooks/           # Your Jupyter notebooks
├── uploads/             # CSV/data files
├── jupyter_lab_config.py # JupyterLab configuration
├── start.sh             # Startup script
└── pyproject.toml       # Dependencies
```

## Tech Stack

- **Frontend**: JupyterLab 4.4+
- **Backend**: Flask + OpenAI API (from v4)
- **AI Models**:
  - GPT-5 nano for code validation
  - GPT-5 mini for error analysis
- **Libraries**: TensorFlow, pandas, matplotlib, plotly, scikit-learn, yfinance

## Setup

```bash
# Already done:
cd /home/slimpunt/0-BRON/0-Nieuwe\ Projecten/projecten/v5JL
poetry install

# Start everything:
./start.sh
```

## URLs

- **JupyterLab**: http://localhost:8888/lab
- **AI Chat**: http://localhost:5001
- **Backend API**: http://localhost:5000

## How to Use

### Quick Start

1. **Start all services:**
   ```bash
   cd /home/slimpunt/0-BRON/0-Nieuwe\ Projecten/projecten/v5JL
   ./start.sh
   ```

2. **Open in browser:**
   - JupyterLab: http://localhost:8888/lab
   - AI Chat: http://localhost:5001

3. **Recommended setup: Split-screen view**
   - Left side: JupyterLab (for notebooks)
   - Right side: AI Chat (for code generation)

### Using AI Chat with JupyterLab

1. **Generate Code:**
   - Use quick prompts or type your request in AI Chat
   - Example: "Generate code to load and analyze a CSV file"
   - AI generates Python code

2. **Copy to JupyterLab:**
   - Click "Copy" button on generated code
   - Paste into JupyterLab cell
   - Run with `Shift+Enter`

3. **Quick Prompts Available:**
   - Load CSV
   - Matplotlib Chart
   - Plotly Chart
   - TensorFlow NN
   - Pandas Analysis
   - Explain Error

### Example Workflow

```
┌─────────────────────────┬─────────────────────────┐
│  JupyterLab             │  AI Chat                │
│  localhost:8888         │  localhost:5001         │
├─────────────────────────┼─────────────────────────┤
│  [Cell 1]               │  You: "Generate code to │
│  # Paste AI code here   │  create a scatter plot  │
│                         │  with matplotlib"       │
│  [Cell 2]               │                         │
│  # More code            │  AI: "Here's code for   │
│                         │  a scatter plot..."     │
│  [Cell 3]               │  [Copy button]          │
│  # Execute & visualize  │                         │
└─────────────────────────┴─────────────────────────┘
```

## Features

### AI Chat Interface
- ✅ Real-time AI code generation
- ✅ Quick prompt buttons
- ✅ Code syntax highlighting
- ✅ One-click copy to clipboard
- ✅ Chat history
- ✅ Backend connection status

### JupyterLab
- ✅ Professional notebook UI
- ✅ Syntax highlighting
- ✅ Code completion
- ✅ Cell execution
- ✅ Rich output (plots, tables, etc.)
- ✅ Multiple notebook support
- ✅ Terminal access

### Backend API
- ✅ AI code generation (GPT-5 nano)
- ✅ Code validation before execution
- ✅ Error analysis and retry (GPT-5 mini)
- ✅ CSV auto-detection
- ✅ TensorFlow, pandas, matplotlib, plotly, scikit-learn

## Notes

- v4 remains untouched and fully functional
- Backend code is copied (can evolve independently)
- Chat UI is separate web app (easy to customize)
- No complex JupyterLab extension needed
- Simple copy-paste workflow
