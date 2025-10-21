const BACKEND_URL = 'http://localhost:5000';
const contentContainer = document.getElementById('contentContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const backendStatus = document.getElementById('backendStatus');
const autoRunToggle = document.getElementById('autoRunToggle');

let cellIdCounter = 0;
let cells = {};
let uploadedFile = null; // Store uploaded file info
let selectionTimer = null;
let selectedCode = '';
let selectedCellId = null;
let selectionContext = null; // 'code', 'output', or 'message'

// Auto-run state - persist in localStorage
let autoRunEnabled = localStorage.getItem('autoRunEnabled') === 'true';
if (autoRunEnabled) {
    autoRunToggle.checked = true;
}

// Handle auto-run toggle change
autoRunToggle.addEventListener('change', function() {
    autoRunEnabled = this.checked;
    localStorage.setItem('autoRunEnabled', autoRunEnabled);
    console.log('Auto-run:', autoRunEnabled ? 'enabled' : 'disabled');
});

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

// Quick prompts - only handle buttons that have a data-prompt attribute
document.querySelectorAll('.quick-prompt-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const prompt = btn.dataset.prompt;
        // Only send message if button has a prompt
        if (prompt) {
            messageInput.value = prompt;
            messageInput.focus();
            sendMessage();
        }
    });
});

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    // Add user message
    addUserMessage(message);
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Show loading
    const loadingId = addLoading();
    sendBtn.disabled = true;

    try {
        // Build request body
        const requestBody = { message };

        // If a file is uploaded, add ONLY the filepath (not the content!)
        if (uploadedFile) {
            requestBody.uploaded_file = {
                filename: uploadedFile.filename,
                filepath: uploadedFile.filepath,
                extension: uploadedFile.extension
            };
        }

        // Collect ALL cells with their code for context
        const recentCells = [];
        Object.keys(cells).forEach(cellId => {
            const cell = cells[cellId];
            if (cell && cell.editor) {
                const code = cell.editor.getValue();
                if (code && code.trim()) {
                    recentCells.push({
                        type: 'code',
                        code: code.trim()
                    });
                }
            }
        });

        // Add cells to request if any exist
        if (recentCells.length > 0) {
            requestBody.recent_cells = recentCells;
        }

        const response = await fetch(`${BACKEND_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();
        removeLoading(loadingId);

        if (data.error) {
            addAssistantMessage(data.error);
        } else if (data.type === 'code') {
            // Add AI message
            addAssistantMessage('Here\'s the code:');
            // Add code cell immediately below
            const cellId = addCodeCell(data.message);

            // Auto-run if enabled
            if (autoRunEnabled && cellId) {
                // Small delay to ensure cell is fully rendered
                setTimeout(() => {
                    console.log('Auto-running cell:', cellId);
                    runCell(cellId);
                }, 300);
            }
        } else {
            addAssistantMessage(data.message || 'No response');
        }
    } catch (error) {
        removeLoading(loadingId);
        addAssistantMessage(`Error: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
    }
}

function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `
        <span class="message-label">></span>
        <span class="message-content">${escapeHtml(text)}</span>
    `;
    contentContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Configure marked.js for syntax highlighting
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try {
                return hljs.highlight(code, { language: lang }).value;
            } catch (err) {
                console.error('Highlight.js error:', err);
            }
        }
        try {
            return hljs.highlightAuto(code).value;
        } catch (err) {
            console.error('Highlight.js auto error:', err);
            return code;
        }
    },
    breaks: true,
    gfm: true
});

async function addAssistantMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    // Render markdown using marked.js with syntax highlighting
    const htmlContent = marked.parse(text);

    messageDiv.innerHTML = `
        <span class="message-content">${htmlContent}</span>
    `;
    contentContainer.appendChild(messageDiv);

    // Add suggestion buttons after the message (async)
    await addChatSuggestions(messageDiv, text);

    scrollToBottom();
}

// Generate AI-powered context-aware suggestions
async function generateSuggestions(message) {
    try {
        // Get ALL cells for context (entire notebook is AI context)
        const recentCells = Object.values(cells).map(cell => ({
            type: 'code',
            code: cell.code || '',
            output: cell.output || ''
        }));

        // Call AI suggestions endpoint
        const response = await fetch(`${BACKEND_URL}/suggestions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                last_message: message,
                recent_cells: recentCells,
                uploaded_file: uploadedFile
            })
        });

        const data = await response.json();

        // Convert AI suggestions to button format
        return (data.suggestions || []).map((label, i) => ({
            label: label,
            type: i === 0 ? 'primary' : '',  // First suggestion is primary
            prompt: label  // Use the suggestion text as the prompt
        }));

    } catch (error) {
        console.error('AI suggestion error:', error);
        // Fallback to basic suggestions
        return [
            { label: 'Continue', type: 'primary', prompt: 'continue with this' },
            { label: 'Next step', type: '', prompt: 'what is the next step?' }
        ];
    }
}

// Add chat suggestions to a message (async)
async function addChatSuggestions(messageDiv, message) {
    // Generate AI suggestions
    const suggestions = await generateSuggestions(message);

    if (suggestions.length === 0) return;

    // Create unique ID for this message's suggestions
    const suggestionId = 'suggestions-' + Date.now();

    // Create suggestion HTML
    const suggestionHTML = `
        <div class="suggestion-container">
            <span class="suggestion-header">
                <span>Next:</span>
            </span>
            <span class="suggestion-buttons" id="${suggestionId}">
                ${suggestions.map((s, i) => `
                    <button class="suggestion-btn ${s.type || ''}" data-suggestion="${i}">
                        <span>${s.label}</span>
                    </button>
                `).join('')}
            </span>
        </div>
    `;

    // Append to message content
    const messageContent = messageDiv.querySelector('.message-content');
    if (messageContent) {
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = suggestionHTML;
        messageContent.appendChild(tempDiv.firstElementChild);

        // Add click handlers
        const btnContainer = document.getElementById(suggestionId);
        if (btnContainer) {
            btnContainer.querySelectorAll('.suggestion-btn').forEach((btn, index) => {
                btn.addEventListener('click', () => {
                    handleSuggestionClick(suggestions[index]);
                });
            });
        }
    }
}

// Handle suggestion button click
function handleSuggestionClick(suggestion) {
    // Set the input value to the suggestion prompt
    messageInput.value = suggestion.prompt;

    // Focus the input
    messageInput.focus();

    // Auto-send the message
    setTimeout(() => {
        sendMessage();
    }, 100);
}

function addLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.id = 'loading-' + Date.now();
    loadingDiv.innerHTML = `
        <span class="message-content">
            <div class="loading">
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
            </div>
        </span>
    `;
    contentContainer.appendChild(loadingDiv);
    scrollToBottom();
    return loadingDiv.id;
}

function removeLoading(id) {
    const loadingDiv = document.getElementById(id);
    if (loadingDiv) loadingDiv.remove();
}

function addCodeCell(code = '') {
    const cellId = 'cell-' + cellIdCounter++;

    const cellDiv = document.createElement('div');
    cellDiv.className = 'code-cell flash';
    cellDiv.id = cellId;

    cellDiv.innerHTML = `
        <div class="cell-toolbar">
            <span class="cell-label">Code Cell [${Object.keys(cells).length + 1}]</span>
            <div class="cell-actions">
                <button class="cell-btn run" onclick="runCell('${cellId}')">▶ Run</button>
                <button class="cell-btn" id="${cellId}-stop-btn" onclick="stopCell('${cellId}')" style="display: none;">⏹ Stop</button>
                <button class="cell-btn" id="${cellId}-expand-btn" onclick="toggleOutputExpansion('${cellId}')" style="display: none;">↕ Expand</button>
                <button class="cell-btn" onclick="clearCellOutput('${cellId}')">Clear</button>
                <button class="cell-btn" onclick="deleteCell('${cellId}')">Delete</button>
            </div>
        </div>
        <div class="cell-input">
            <textarea id="${cellId}-code"></textarea>
        </div>
    `;

    contentContainer.appendChild(cellDiv);

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

    // Add selection handler for "Explain" button
    let lastSelection = '';

    editor.on('cursorActivity', (cm) => {
        // Clear existing timer
        if (selectionTimer) {
            clearTimeout(selectionTimer);
        }

        const selection = cm.getSelection();

        if (selection && selection.trim().length > 0) {
            // Store selection immediately
            lastSelection = selection;
            selectedCode = selection;
            selectedCellId = cellId;

            // User has selected code - show button after delay
            selectionTimer = setTimeout(() => {
                // Use the last stored selection, not current (might be empty now)
                if (lastSelection) {
                    showExplainButton(cm, cellId, lastSelection);
                }
            }, 500); // 500ms delay
        } else {
            // No selection - hide button after a delay if it's visible
            const explainBtn = document.getElementById('explainBtn');
            if (explainBtn && explainBtn.style.display === 'block') {
                // Give user time to move mouse to button
                setTimeout(() => {
                    // Check if button is still not being hovered
                    const explainBtn = document.getElementById('explainBtn');
                    if (explainBtn && explainBtn.style.display === 'block' && !explainBtn.matches(':hover')) {
                        lastSelection = '';
                        selectedCode = '';
                        hideExplainButton();
                    }
                }, 1000); // 1 second delay to move to button
            } else {
                // Button not visible, clear selection
                lastSelection = '';
                selectedCode = '';
            }
        }
    });

    if (code) {
        editor.setValue(code);
    }

    cells[cellId] = {
        editor: editor,
        outputDiv: null
    };

    // Focus new cell
    editor.focus();
    scrollToBottom();

    // Remove flash after animation
    setTimeout(() => {
        cellDiv.classList.remove('flash');
    }, 1000);

    return cellId;
}

async function runCell(cellId) {
    const cell = cells[cellId];
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

    // Show stop button in toolbar
    const stopBtn = document.getElementById(`${cellId}-stop-btn`);
    if (stopBtn) stopBtn.style.display = 'inline-block';

    scrollToBottom();

    // Create AbortController for this execution
    const abortController = new AbortController();
    cell.abortController = abortController;

    try {
        const response = await fetch(`${BACKEND_URL}/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code }),
            signal: abortController.signal
        });

        const data = await response.json();

        // Check for error
        if (data.error) {
            outputDiv.className = 'cell-output error';
            // Handle both string and object error formats
            const errorText = typeof data.error === 'string'
                ? data.error
                : (data.error.technical || data.error.explanation || 'Execution error');

            // Store error in cell for AI debugging
            cell.lastError = errorText;

            // Create error display with AI fix button
            outputDiv.innerHTML = `
                <div style="margin-bottom: 10px;">${escapeHtml(errorText)}</div>
                <button onclick="askAIToFix('${cellId}')" style="background: #569cd6; color: #1e1e1e; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600;">
                    🤖 Ask AI to Fix
                </button>
            `;
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
                    output += `<div id="${plotId}" style="width: 100%; height: 900px; margin: 10px 0;"></div>`;

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
                // Show expand button when output is displayed
                const expandBtn = document.getElementById(`${cellId}-expand-btn`);
                if (expandBtn) {
                    expandBtn.style.display = 'inline-block';
                }
            } else {
                outputDiv.remove();
                cell.outputDiv = null;
                // Hide expand button when no output
                const expandBtn = document.getElementById(`${cellId}-expand-btn`);
                if (expandBtn) {
                    expandBtn.style.display = 'none';
                }
            }
        }

        scrollToBottom();
    } catch (error) {
        // Check if execution was aborted
        if (error.name === 'AbortError') {
            outputDiv.className = 'cell-output';
            outputDiv.textContent = '⏹ Execution stopped by user';
            outputDiv.style.color = '#f48771';
        } else {
            outputDiv.className = 'cell-output error';
            outputDiv.textContent = `Error: ${error.message}`;
        }
        // Show expand button even for errors
        const expandBtn = document.getElementById(`${cellId}-expand-btn`);
        if (expandBtn) {
            expandBtn.style.display = 'inline-block';
        }
    } finally {
        // Hide stop button
        const stopBtn = document.getElementById(`${cellId}-stop-btn`);
        if (stopBtn) stopBtn.style.display = 'none';

        // Clean up abort controller
        if (cell.abortController) {
            cell.abortController = null;
        }
    }
}

function stopCell(cellId) {
    const cell = cells[cellId];
    if (cell && cell.abortController) {
        cell.abortController.abort();
    }
}

function clearCellOutput(cellId) {
    const cell = cells[cellId];
    if (cell && cell.outputDiv) {
        cell.outputDiv.remove();
        cell.outputDiv = null;
        // Hide expand button when output is cleared
        const expandBtn = document.getElementById(`${cellId}-expand-btn`);
        if (expandBtn) {
            expandBtn.style.display = 'none';
        }
    }
}

function deleteCell(cellId) {
    const cell = cells[cellId];
    if (!cell) return;

    const cellDiv = document.getElementById(cellId);
    if (cellDiv) cellDiv.remove();

    delete cells[cellId];
}

function toggleOutputExpansion(cellId) {
    const cell = cells[cellId];
    if (!cell || !cell.outputDiv) return;

    const expandBtn = document.getElementById(`${cellId}-expand-btn`);

    if (cell.outputDiv.classList.contains('expanded')) {
        // Collapse output
        cell.outputDiv.classList.remove('expanded');
        expandBtn.textContent = '↕ Expand';
    } else {
        // Expand output
        cell.outputDiv.classList.add('expanded');
        expandBtn.textContent = '↕ Compact';
    }
}

function scrollToBottom() {
    window.scrollTo({
        top: document.body.scrollHeight,
        behavior: 'smooth'
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Toggle functions
let chatHidden = false;
let codeHidden = false;

function toggleChat() {
    chatHidden = !chatHidden;
    const messages = document.querySelectorAll('.message');
    const btn = document.getElementById('toggleChatBtn');

    messages.forEach(msg => {
        if (chatHidden) {
            msg.classList.add('hidden');
        } else {
            msg.classList.remove('hidden');
        }
    });

    btn.textContent = chatHidden ? 'Show Chat' : 'Hide Chat';
    btn.classList.toggle('active', chatHidden);
}

function toggleCode() {
    codeHidden = !codeHidden;
    const cells = document.querySelectorAll('.code-cell');
    const btn = document.getElementById('toggleCodeBtn');

    cells.forEach(cell => {
        const input = cell.querySelector('.cell-input');
        const toolbar = cell.querySelector('.cell-toolbar');

        if (codeHidden) {
            input.classList.add('hidden');
            cell.classList.add('code-hidden');
        } else {
            input.classList.remove('hidden');
            cell.classList.remove('code-hidden');
        }
    });

    btn.textContent = codeHidden ? 'Show Code' : 'Hide Code';
    btn.classList.toggle('active', codeHidden);
}

// File upload handler
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Show loading message
    addAssistantMessage(`Uploading ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${BACKEND_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            addAssistantMessage(`Upload failed: ${data.error}`);
            return;
        }

        // Store uploaded file info
        uploadedFile = {
            filename: data.filename,
            filepath: data.filepath,
            extension: data.extension
        };

        // Show success message
        addAssistantMessage(`File uploaded: ${data.filename}\n\nI can now generate code to analyze this file. Just ask me!`);

        // Reset file input
        event.target.value = '';

    } catch (error) {
        addAssistantMessage(`Upload error: ${error.message}`);
    }
}

// Code selection "Explain" button functions
function showExplainButton(editor, cellId, code) {
    const explainBtn = document.getElementById('explainBtn');
    selectedCode = code;
    selectedCellId = cellId;
    selectionContext = 'code';

    // Get cursor position
    const cursor = editor.getCursor();
    const coords = editor.cursorCoords(cursor, 'page');

    // Position button below selection
    explainBtn.style.left = coords.left + 'px';
    explainBtn.style.top = (coords.top + 25) + 'px';
    explainBtn.style.display = 'block';
}

function hideExplainButton() {
    const explainBtn = document.getElementById('explainBtn');
    explainBtn.style.display = 'none';
    selectedCode = '';
    selectedCellId = null;
    selectionContext = null;
}

async function explainSelectedCode() {
    if (!selectedCode) return;

    // Store text and context in local variables BEFORE hiding button (which clears them)
    const textToExplain = selectedCode;
    const context = selectionContext;
    const cellId = selectedCellId;

    hideExplainButton();

    // Create context-aware prompt
    let userMessage = '';
    let aiMessage = '';

    if (context === 'code') {
        // Explaining code
        userMessage = `Explain this code:\n\n\`\`\`python\n${textToExplain}\n\`\`\``;
        aiMessage = `Explain this Python code in detail:\n\n${textToExplain}`;
    } else if (context === 'output') {
        // Explaining cell output
        let contextCode = '';
        if (cellId && cells[cellId] && cells[cellId].editor) {
            contextCode = cells[cellId].editor.getValue();
        }

        if (contextCode) {
            userMessage = `Explain this output:\n\n${textToExplain}\n\nFrom executing:\n\`\`\`python\n${contextCode}\n\`\`\``;
            aiMessage = `Explain this output:\n\n${textToExplain}\n\nThis output came from executing the following Python code:\n\n${contextCode}`;
        } else {
            userMessage = `Explain this output:\n\n${textToExplain}`;
            aiMessage = `Explain this output:\n\n${textToExplain}`;
        }
    } else if (context === 'message') {
        // Explaining AI message
        userMessage = `Explain this:\n\n${textToExplain}`;
        aiMessage = `Please explain this in more detail:\n\n${textToExplain}`;
    } else {
        // Fallback
        userMessage = `Explain this:\n\n${textToExplain}`;
        aiMessage = `Please explain this:\n\n${textToExplain}`;
    }

    // Add user message
    addUserMessage(userMessage);

    // Show loading
    const loadingId = addLoading();
    sendBtn.disabled = true;

    try {
        const response = await fetch(`${BACKEND_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: aiMessage
            })
        });

        const data = await response.json();
        removeLoading(loadingId);

        if (data.error) {
            addAssistantMessage(data.error);
        } else {
            addAssistantMessage(data.message || 'No response');
        }
    } catch (error) {
        removeLoading(loadingId);
        addAssistantMessage(`Error: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
    }
}

// Event listeners
sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Explain button click handler
document.getElementById('explainBtn').addEventListener('click', explainSelectedCode);

// Hide explain button when clicking outside
document.addEventListener('click', (e) => {
    const explainBtn = document.getElementById('explainBtn');
    if (e.target !== explainBtn && !e.target.closest('.CodeMirror')) {
        hideExplainButton();
    }
});

// Global text selection handler for cell outputs and AI messages
document.addEventListener('mouseup', (e) => {
    // Skip if clicking on CodeMirror (handled separately)
    if (e.target.closest('.CodeMirror')) {
        return;
    }

    // Skip if clicking the explain button itself
    const explainBtn = document.getElementById('explainBtn');
    if (e.target === explainBtn || e.target.closest('#explainBtn')) {
        return;
    }

    // Clear existing timer
    if (selectionTimer) {
        clearTimeout(selectionTimer);
    }

    // Small delay to allow selection to finalize
    selectionTimer = setTimeout(() => {
        const selection = window.getSelection();
        const selectedText = selection.toString().trim();

        if (!selectedText || selectedText.length === 0) {
            // No selection - hide button if visible
            if (explainBtn && explainBtn.style.display === 'block') {
                setTimeout(() => {
                    if (explainBtn && explainBtn.style.display === 'block' && !explainBtn.matches(':hover')) {
                        hideExplainButton();
                    }
                }, 1000);
            }
            return;
        }

        // Check what type of content was selected
        const range = selection.getRangeAt(0);
        const container = range.commonAncestorContainer;
        const parentElement = container.nodeType === Node.TEXT_NODE ? container.parentElement : container;

        // Check if selection is in a cell output
        const cellOutput = parentElement.closest('.cell-output');
        if (cellOutput) {
            // Find the cell ID
            const cellDiv = cellOutput.closest('.code-cell');
            const cellId = cellDiv ? cellDiv.id : null;

            selectedCode = selectedText;
            selectedCellId = cellId;
            selectionContext = 'output';

            // Position button near selection
            const rect = range.getBoundingClientRect();
            explainBtn.style.left = rect.left + 'px';
            explainBtn.style.top = (rect.bottom + 5) + 'px';
            explainBtn.style.display = 'block';
            return;
        }

        // Check if selection is in an AI message
        const messageContent = parentElement.closest('.message-content');
        if (messageContent) {
            selectedCode = selectedText;
            selectedCellId = null;
            selectionContext = 'message';

            // Position button near selection
            const rect = range.getBoundingClientRect();
            explainBtn.style.left = rect.left + 'px';
            explainBtn.style.top = (rect.bottom + 5) + 'px';
            explainBtn.style.display = 'block';
            return;
        }

        // No relevant content selected
        hideExplainButton();
    }, 100);
});

// ===== IMPORT/EXPORT FUNCTIONALITY =====

function showExportMenu() {
    const format = confirm("Export as .ipynb (OK) or .py (Cancel)?");
    if (format) {
        exportNotebook('ipynb');
    } else {
        exportNotebook('py');
    }
}

async function exportNotebook(format) {
    try {
        // Collect all cells with code and output
        const cellsData = [];
        Object.keys(cells).forEach(cellId => {
            const cell = cells[cellId];
            if (cell && cell.editor) {
                const code = cell.editor.getValue();
                const output = cell.outputDiv ? cell.outputDiv.textContent : '';
                cellsData.push({ code, output });
            }
        });

        const response = await fetch(`${BACKEND_URL}/export/${format}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cells: cellsData })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = format === 'ipynb' ? 'notebook.ipynb' : 'notebook.py';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        } else {
            alert('Export failed');
        }
    } catch (error) {
        alert(`Export error: ${error.message}`);
    }
}

async function handleNotebookImport(event) {
    const file = event.target.files[0];
    if (!file) return;

    const fileName = file.name;
    const fileExt = fileName.split('.').pop().toLowerCase();

    try {
        const content = await file.text();

        const response = await fetch(`${BACKEND_URL}/import/${fileExt}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content })
        });

        const data = await response.json();

        if (data.error) {
            alert(`Import error: ${data.error}`);
            return;
        }

        // Clear existing cells
        Object.keys(cells).forEach(cellId => {
            const cellDiv = document.getElementById(cellId);
            if (cellDiv) cellDiv.remove();
        });
        cells = {};

        // Create new cells from imported data
        data.cells.forEach(cellData => {
            const cellId = addCodeCell(cellData.code);

            // If cell has output, display it
            if (cellData.output) {
                const cell = cells[cellId];
                if (cell) {
                    const cellDiv = document.getElementById(cellId);
                    const outputDiv = document.createElement('div');
                    outputDiv.className = 'cell-output';
                    outputDiv.textContent = cellData.output;
                    cellDiv.appendChild(outputDiv);
                    cell.outputDiv = outputDiv;
                }
            }
        });

        alert(`Successfully imported ${data.cells.length} cell(s) from ${fileName}`);
    } catch (error) {
        alert(`Import error: ${error.message}`);
    }

    // Reset file input
    event.target.value = '';
}

// ===== KERNEL RESTART FUNCTIONALITY =====

async function restartKernel() {
    // Show confirmation dialog
    if (!confirm('Are you sure you want to restart the kernel?\n\nThis will clear all variables and state.')) {
        return;
    }

    try {
        const response = await fetch(`${BACKEND_URL}/restart_kernel`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.status === 'ok') {
            // Show success message
            addAssistantMessage('Kernel restarted - all variables cleared');
        } else {
            addAssistantMessage(`Kernel restart failed: ${data.message}`);
        }
    } catch (error) {
        addAssistantMessage(`Kernel restart error: ${error.message}`);
    }
}

// Ask AI to fix an error
async function askAIToFix(cellId) {
    const cell = cells[cellId];
    if (!cell || !cell.editor || !cell.lastError) return;

    const code = cell.editor.getValue();
    const errorMessage = cell.lastError;

    // Construct AI prompt
    const message = `Fix this error:\n\nCode:\n\`\`\`python\n${code}\n\`\`\`\n\nError:\n${errorMessage}\n\nPlease fix the code and explain what was wrong.`;

    // Set the input and trigger send
    messageInput.value = message;
    sendMessage();
}
