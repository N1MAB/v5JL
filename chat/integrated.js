const BACKEND_URL = 'http://localhost:5000';
const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const backendStatus = document.getElementById('backendStatus');
const cellsContainer = document.getElementById('cellsContainer');

let cells = [];
let cellIdCounter = 0;

// Check backend status
async function checkBackend() {
    try {
        const response = await fetch(`${BACKEND_URL}/health`);
        const data = await response.json();
        backendStatus.textContent = '✓ Connected';
        backendStatus.style.color = '#4ec9b0';
    } catch (error) {
        backendStatus.textContent = '✗ Disconnected';
        backendStatus.style.color = '#f48771';
    }
}

checkBackend();
setInterval(checkBackend, 30000);

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 80) + 'px';
});

// Quick prompts
document.querySelectorAll('.quick-prompt-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const prompt = btn.dataset.prompt;
        messageInput.value = prompt;
        messageInput.focus();
        sendMessage();
    });
});

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    addMessage('user', message);
    messageInput.value = '';
    messageInput.style.height = 'auto';

    const loadingId = addLoading();
    sendBtn.disabled = true;

    try {
        const response = await fetch(`${BACKEND_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        removeLoading(loadingId);

        if (data.error) {
            addMessage('assistant', data.error);
        } else if (data.type === 'code') {
            addCodeMessage(data.message);
        } else {
            addMessage('assistant', data.message || 'No response');
        }
    } catch (error) {
        removeLoading(loadingId);
        addMessage('assistant', `Error: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
    }
}

function addMessage(role, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const label = role === 'user' ? 'You' : 'AI Assistant';
    messageDiv.innerHTML = `
        <div>
            <div class="message-label">${label}</div>
            <div class="message-content">${escapeHtml(text)}</div>
        </div>
    `;

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addCodeMessage(code) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    messageDiv.innerHTML = `
        <div>
            <div class="message-label">AI Assistant</div>
            <div class="message-content">
                Here's the code:
                <div class="code-preview">
                    <div class="code-header">
                        <span style="color: #4ec9b0; font-size: 11px;">PYTHON</span>
                        <div class="code-actions">
                            <button class="code-btn" onclick="copyCode(this, \`${escapeForAttribute(code)}\`)">Copy</button>
                            <button class="code-btn primary" onclick="addCodeToNewCell(\`${escapeForAttribute(code)}\`)">➕ Add to Cell</button>
                        </div>
                    </div>
                    <pre><code>${escapeHtml(code)}</code></pre>
                </div>
            </div>
        </div>
    `;

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.id = 'loading-' + Date.now();
    loadingDiv.innerHTML = `
        <div>
            <div class="message-label">AI Assistant</div>
            <div class="message-content">
                <div class="loading">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
            </div>
        </div>
    `;
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return loadingDiv.id;
}

function removeLoading(id) {
    const loadingDiv = document.getElementById(id);
    if (loadingDiv) loadingDiv.remove();
}

function copyCode(btn, code) {
    navigator.clipboard.writeText(code);
    const originalText = btn.textContent;
    btn.textContent = '✓ Copied!';
    setTimeout(() => {
        btn.textContent = originalText;
    }, 2000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeForAttribute(text) {
    return text.replace(/`/g, '\\`').replace(/\$/g, '\\$').replace(/\\/g, '\\\\');
}

// Event listeners
sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// ===== NOTEBOOK FUNCTIONALITY =====

function addNewCell(code = '') {
    const cellId = 'cell-' + cellIdCounter++;

    const cellDiv = document.createElement('div');
    cellDiv.className = 'cell';
    cellDiv.id = cellId;

    cellDiv.innerHTML = `
        <div class="cell-toolbar">
            <span class="cell-label">Cell [${cells.length + 1}]</span>
            <div class="cell-actions">
                <button class="cell-btn run" onclick="runCell('${cellId}')">▶ Run</button>
                <button class="cell-btn" onclick="clearCellOutput('${cellId}')">Clear</button>
                <button class="cell-btn" onclick="deleteCell('${cellId}')">Delete</button>
            </div>
        </div>
        <div class="cell-input">
            <textarea id="${cellId}-code"></textarea>
        </div>
    `;

    cellsContainer.appendChild(cellDiv);

    // Initialize CodeMirror
    const textarea = document.getElementById(`${cellId}-code`);
    const editor = CodeMirror.fromTextArea(textarea, {
        mode: 'python',
        theme: 'monokai',
        lineNumbers: true,
        indentUnit: 4,
        lineWrapping: true,
        extraKeys: {
            'Shift-Enter': () => runCell(cellId),
            'Tab': (cm) => {
                if (cm.somethingSelected()) {
                    cm.indentSelection('add');
                } else {
                    cm.replaceSelection('    ', 'end');
                }
            }
        }
    });

    if (code) {
        editor.setValue(code);
    }

    cells.push({
        id: cellId,
        editor: editor,
        outputDiv: null
    });

    // Focus new cell
    editor.focus();

    // Scroll to new cell
    cellDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    return cellId;
}

function addCodeToNewCell(code) {
    const cellId = addNewCell(code);
    const cell = cells.find(c => c.id === cellId);
    if (cell) {
        // Flash the cell to show it was added
        const cellDiv = document.getElementById(cellId);
        cellDiv.style.borderColor = '#4ec9b0';
        setTimeout(() => {
            cellDiv.style.borderColor = '';
        }, 1000);
    }
}

async function runCell(cellId) {
    const cell = cells.find(c => c.id === cellId);
    if (!cell) return;

    const code = cell.editor.getValue().trim();
    if (!code) return;

    // Clear previous output
    if (cell.outputDiv) {
        cell.outputDiv.remove();
        cell.outputDiv = null;
    }

    // Create output div
    const cellDiv = document.getElementById(cellId);
    const outputDiv = document.createElement('div');
    outputDiv.className = 'cell-output';
    outputDiv.innerHTML = '<div class="loading"><div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div></div>';
    cellDiv.appendChild(outputDiv);
    cell.outputDiv = outputDiv;

    try {
        const response = await fetch(`${BACKEND_URL}/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        });

        const data = await response.json();

        // Check for error
        if (data.error) {
            outputDiv.className = 'cell-output error';
            // Handle both string and object error formats
            if (typeof data.error === 'string') {
                outputDiv.textContent = data.error;
            } else {
                outputDiv.textContent = data.error.technical || data.error.explanation || 'Execution error';
            }
        } else {
            // Success! Build output HTML
            let output = '';

            // Text output
            if (data.output) {
                output += `<pre>${escapeHtml(data.output)}</pre>`;
            }

            // Matplotlib plots
            if (data.plots && data.plots.length > 0) {
                data.plots.forEach(img => {
                    output += `<img src="data:image/png;base64,${img}" alt="Plot" style="max-width: 100%; margin: 10px 0;">`;
                });
            }

            // Plotly interactive plots
            if (data.plotly_data && data.plotly_data.length > 0) {
                data.plotly_data.forEach((plotData, index) => {
                    const plotId = `plot-${cellId}-${index}`;
                    output += `<div id="${plotId}" style="width: 100%; height: 500px; margin: 10px 0;"></div>`;

                    // Load Plotly and render after DOM update
                    setTimeout(() => {
                        if (typeof Plotly === 'undefined') {
                            // Load Plotly dynamically
                            const script = document.createElement('script');
                            script.src = 'https://cdn.plot.ly/plotly-2.26.0.min.js';
                            script.onload = () => {
                                Plotly.newPlot(plotId, JSON.parse(plotData).data, JSON.parse(plotData).layout);
                            };
                            document.head.appendChild(script);
                        } else {
                            Plotly.newPlot(plotId, JSON.parse(plotData).data, JSON.parse(plotData).layout);
                        }
                    }, 100);
                });
            }

            // Display output or remove if empty
            if (output) {
                outputDiv.innerHTML = output;
            } else {
                outputDiv.remove();
                cell.outputDiv = null;
            }
        }
    } catch (error) {
        outputDiv.className = 'cell-output error';
        outputDiv.textContent = `Error: ${error.message}`;
    }
}

function clearCellOutput(cellId) {
    const cell = cells.find(c => c.id === cellId);
    if (cell && cell.outputDiv) {
        cell.outputDiv.remove();
        cell.outputDiv = null;
    }
}

function deleteCell(cellId) {
    const cellIndex = cells.findIndex(c => c.id === cellId);
    if (cellIndex === -1) return;

    // Remove from DOM
    const cellDiv = document.getElementById(cellId);
    if (cellDiv) cellDiv.remove();

    // Remove from array
    cells.splice(cellIndex, 1);

    // Update cell labels
    cells.forEach((cell, idx) => {
        const labelSpan = document.querySelector(`#${cell.id} .cell-label`);
        if (labelSpan) {
            labelSpan.textContent = `Cell [${idx + 1}]`;
        }
    });
}

function runAllCells() {
    cells.forEach(cell => runCell(cell.id));
}

function clearAllOutputs() {
    cells.forEach(cell => clearCellOutput(cell.id));
}

// Add first cell on load
window.addEventListener('load', () => {
    addNewCell();
});

// ===== RESIZE FUNCTIONALITY =====
const resizeHandle = document.getElementById('resizeHandle');
const chatSection = document.querySelector('.chat-section');
const mainContainer = document.querySelector('.main-container');

let isResizing = false;

resizeHandle.addEventListener('mousedown', (e) => {
    isResizing = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
});

document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;

    const containerRect = mainContainer.getBoundingClientRect();
    const newWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;

    // Constrain between 20% and 70%
    if (newWidth >= 20 && newWidth <= 70) {
        chatSection.style.flex = `0 0 ${newWidth}%`;
    }
});

document.addEventListener('mouseup', () => {
    if (isResizing) {
        isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    }
});
