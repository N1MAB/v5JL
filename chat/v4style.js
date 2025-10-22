const BACKEND_URL = 'http://localhost:5000';
const contentContainer = document.getElementById('contentContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const backendStatus = document.getElementById('backendStatus');
const autoRunToggle = document.getElementById('autoRunToggle');

// Generate unique session ID for this browser tab
function generateSessionId() {
    return 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
}

const SESSION_ID = generateSessionId();
console.log('Session ID:', SESSION_ID);

let cellIdCounter = 0;
let cells = {};
let uploadedFile = null; // Will store: {filename, fileData (base64), fileType, fileSize}
let selectionTimer = null;
let selectedCode = '';
let selectedCellId = null;
let selectionContext = null; // 'code', 'output', or 'message'

// Insert new cells after this cellId (null = insert at end)
let insertAfterCellId = null;

// Auto-run state - persist in localStorage
let autoRunEnabled = localStorage.getItem('autoRunEnabled') === 'true';
if (autoRunEnabled) {
    autoRunToggle.checked = true;
}

// Context tracking for smart insertion
let lastErrorCellId = null; // Track which cell had the last error
let lastExecutedCellId = null; // Track which cell was executed last
let chatHistory = []; // Track full conversation history

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

    // Smart context detection: if message is about errors/output and insertAfterCellId not set, auto-detect
    if (!insertAfterCellId) {
        const messageLower = message.toLowerCase();
        const errorKeywords = ['error', 'fout', 'waarom', 'probleem', 'werkt niet', 'fix', 'onjuist'];
        const outputKeywords = ['output', 'resultaat', 'uitleg', 'leg uit', 'betekent', 'geeft'];

        // Check if message is about errors
        const isAboutError = errorKeywords.some(keyword => messageLower.includes(keyword));
        // Check if message is about output
        const isAboutOutput = outputKeywords.some(keyword => messageLower.includes(keyword));

        if (isAboutError && lastErrorCellId) {
            insertAfterCellId = lastErrorCellId;
            console.log('Smart detection: Detected error-related question, inserting after error cell');
        } else if (isAboutOutput && lastExecutedCellId) {
            insertAfterCellId = lastExecutedCellId;
            console.log('Smart detection: Detected output-related question, inserting after last executed cell');
        }
    }

    // Add user message
    addUserMessage(message);

    // Track in chat history
    chatHistory.push({ role: 'user', content: message });

    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Show loading
    const loadingId = addLoading();
    sendBtn.disabled = true;

    try {
        // Build request body
        const requestBody = { message };

        // If a file is uploaded, add metadata (NOT file data - too large for chat)
        if (uploadedFile) {
            requestBody.uploaded_file = {
                filename: uploadedFile.filename,
                extension: uploadedFile.extension,
                fileSize: uploadedFile.fileSize
            };
        }

        // Collect ALL cells with their code and execution metadata for FULL context awareness
        const recentCells = [];
        Object.keys(cells).forEach(cellId => {
            const cell = cells[cellId];
            if (cell && cell.editor) {
                const code = cell.editor.getValue();
                if (code && code.trim()) {
                    recentCells.push({
                        type: 'code',
                        code: code.trim(),
                        output: cell.lastOutput || '',  // AI can see execution results
                        hasError: cell.hadError || false,  // Did this cell have an error?
                        executed: cell.executed || false,  // Has this cell been run?
                        cellId: cellId  // Cell reference for debugging
                    });
                }
            }
        });

        // Add cells to request if any exist
        if (recentCells.length > 0) {
            requestBody.recent_cells = recentCells;
        }

        // Add chat history (last 10 messages for context)
        if (chatHistory.length > 0) {
            requestBody.chat_history = chatHistory.slice(-10);
        }

        const response = await fetch(`${BACKEND_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-ID': SESSION_ID // Session isolation
            },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();
        removeLoading(loadingId);

        if (data.error) {
            addAssistantMessage(data.error);
            chatHistory.push({ role: 'assistant', content: data.error });
        } else if (data.type === 'code') {
            // Add AI message
            const msg = 'Here\'s the code:';
            addAssistantMessage(msg);
            chatHistory.push({ role: 'assistant', content: msg + '\n```python\n' + data.message + '\n```' });

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
            const msg = data.message || 'No response';
            addAssistantMessage(msg);
            chatHistory.push({ role: 'assistant', content: msg });
        }
    } catch (error) {
        removeLoading(loadingId);
        const errorMsg = `Error: ${error.message}`;
        addAssistantMessage(errorMsg);
        chatHistory.push({ role: 'assistant', content: errorMsg });
    } finally {
        sendBtn.disabled = false;
        // Reset insert position after message is sent
        insertAfterCellId = null;
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
    // Don't auto-scroll - let user stay at current position
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
    // Split text by code blocks, alternating between text and code
    const parts = [];
    let lastIndex = 0;
    const codeBlockRegex = /```python\n([\s\S]*?)```/g;
    let match;

    while ((match = codeBlockRegex.exec(text)) !== null) {
        // Add text before code block
        if (match.index > lastIndex) {
            parts.push({
                type: 'text',
                content: text.substring(lastIndex, match.index)
            });
        }
        // Add code block
        parts.push({
            type: 'code',
            content: match[1].trim()
        });
        lastIndex = match.index + match[0].length;
    }

    // Add remaining text after last code block
    if (lastIndex < text.length) {
        parts.push({
            type: 'text',
            content: text.substring(lastIndex)
        });
    }

    // If no code blocks found, just add as text
    if (parts.length === 0) {
        parts.push({
            type: 'text',
            content: text
        });
    }

    // Add parts in order: text, code, text, code, etc.
    // Track newly created cell IDs for auto-run
    let lastTextDiv = null;
    const newCellIds = [];

    for (const part of parts) {
        if (part.type === 'text' && part.content.trim()) {
            // Add text message
            const textDiv = document.createElement('div');
            textDiv.className = 'message assistant';
            const htmlContent = marked.parse(part.content);
            textDiv.innerHTML = `<span class="message-content">${htmlContent}</span>`;

            // Insert text: either after a specific cell (if insertAfterCellId is set) or at the end
            if (insertAfterCellId && document.getElementById(insertAfterCellId)) {
                const afterCell = document.getElementById(insertAfterCellId);
                // Insert after the target cell
                afterCell.insertAdjacentElement('afterend', textDiv);
                // Update insertAfterCellId to point to this text div for next items
                textDiv.id = 'text-' + Date.now();
                insertAfterCellId = textDiv.id;
            } else {
                // Default: append at end
                contentContainer.appendChild(textDiv);
            }
            lastTextDiv = textDiv;
        } else if (part.type === 'code') {
            // Add code cell and track its ID
            const cellId = addCodeCell(part.content);
            newCellIds.push(cellId);

            // Update insertAfterCellId to this cell so next items come after it
            if (insertAfterCellId) {
                insertAfterCellId = cellId;
            }
        }
    }

    // Add smart suggestions after last text div
    if (lastTextDiv) {
        await addChatSuggestions(lastTextDiv, text);
    }

    // Auto-run newly created code cells sequentially (top to bottom)
    if (newCellIds.length > 0) {
        for (const cellId of newCellIds) {
            // Run cell and wait for completion
            await runCell(cellId);

            // Check if cell had an error - if so, STOP auto-running
            const cell = cells[cellId];
            if (cell && cell.hadError) {
                console.log(`Auto-run stopped at cell ${cellId} due to error`);
                break; // Stop running remaining cells
            }

            // Small delay between cells
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    // Reset insertAfterCellId after adding all content
    insertAfterCellId = null;

    // Don't auto-scroll - let user scroll manually to see new content
}

// Generate smart static suggestions based on context
async function generateSuggestions(message) {
    const msg = message.toLowerCase();

    // Check if it's a tutorial/step-by-step explanation (text with code blocks)
    const isTutorial = msg.includes('```python') && (
        msg.includes('let\'s') || msg.includes('first') || msg.includes('next') ||
        msg.includes('now') || msg.includes('step') || msg.includes('stap')
    );

    if (isTutorial) {
        return [
            { label: 'Continue tutorial', prompt: 'Continue with next steps', type: 'primary' },
            { label: 'Explain more', prompt: 'Explain this in more detail', type: '' },
            { label: 'Different approach', prompt: 'Show a different way to do this', type: '' }
        ];
    }

    // Check if it's an explanation/text response (no code)
    if (msg.includes('explanation') || msg.includes('you can') || msg.includes('this is') ||
        msg.includes('geweldig') || msg.includes('stap voor stap')) {
        return [
            { label: 'Show code example', prompt: 'Show me a code example', type: 'primary' },
            { label: 'More details', prompt: 'Tell me more', type: '' }
        ];
    }

    // Default suggestions for greetings or general responses
    if (msg.includes('hello') || msg.includes('help') || msg.includes('dataset')) {
        return [
            { label: 'Load CSV', prompt: 'Load CSV file', type: 'primary' },
            { label: 'Example: Iris', prompt: 'Load iris dataset', type: '' },
            { label: 'Example: Titanic', prompt: 'Load titanic dataset', type: '' }
        ];
    }

    // Generic follow-up suggestions
    return [
        { label: 'Visualize', prompt: 'Visualize this data', type: 'primary' },
        { label: 'Statistics', prompt: 'Show statistics', type: '' },
        { label: 'More info', prompt: 'Tell me more', type: '' }
    ];
}

// Add chat suggestions to a message (async)
async function addChatSuggestions(messageDiv, message) {
    // Generate smart suggestions
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
    // Don't auto-scroll - let user stay at current position
    return loadingDiv.id;
}

function removeLoading(id) {
    const loadingDiv = document.getElementById(id);
    if (loadingDiv) loadingDiv.remove();
}

// Create inline executable code cell for AI chat responses
function createInlineChatCodeCell(code = '') {
    const cellId = 'cell-' + cellIdCounter++;

    const cellDiv = document.createElement('div');
    cellDiv.className = 'code-cell chat-code-cell';
    cellDiv.id = cellId;

    cellDiv.innerHTML = `
        <div class="cell-toolbar">
            <span class="cell-label">AI Example [${Object.keys(cells).length + 1}]</span>
            <div class="cell-actions">
                <button class="cell-btn run" onclick="runCell('${cellId}')">▶ Run</button>
                <button class="cell-btn" id="${cellId}-stop-btn" onclick="stopCell('${cellId}')" style="display: none;">⏹ Stop</button>
                <button class="cell-btn" id="${cellId}-expand-btn" onclick="toggleOutputExpansion('${cellId}')" style="display: none;">↕ Expand</button>
                <button class="cell-btn" onclick="clearCellOutput('${cellId}')">Clear</button>
            </div>
        </div>
        <div class="cell-input">
            <textarea id="${cellId}-code"></textarea>
        </div>
    `;

    // Initialize CodeMirror
    const textarea = cellDiv.querySelector('textarea');
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

    return cellDiv;
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
                <button class="cell-btn" onclick="askAboutCell('${cellId}')">💬 Ask</button>
                <button class="cell-btn" onclick="clearCellOutput('${cellId}')">Clear</button>
                <button class="cell-btn" onclick="deleteCell('${cellId}')">Delete</button>
            </div>
        </div>
        <div class="cell-input">
            <textarea id="${cellId}-code"></textarea>
        </div>
    `;

    // Insert cell: either after a specific cell (if insertAfterCellId is set) or at the end
    if (insertAfterCellId && document.getElementById(insertAfterCellId)) {
        const afterCell = document.getElementById(insertAfterCellId);
        // Insert after the target cell (can be in middle of content)
        afterCell.insertAdjacentElement('afterend', cellDiv);
    } else {
        // Default: append at end
        contentContainer.appendChild(cellDiv);
    }

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

    // Don't auto-scroll or focus - let user stay where they are reading
    // editor.focus();

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

    // Don't auto-scroll - let user stay at current position

    // Create AbortController for this execution
    const abortController = new AbortController();
    cell.abortController = abortController;

    try {
        // Prepare request body with code and file data (if uploaded)
        const requestBody = { code };

        // If file is uploaded, include file data for execution
        if (uploadedFile) {
            requestBody.fileData = uploadedFile.fileData; // base64
            requestBody.fileName = uploadedFile.filename;
            requestBody.fileType = uploadedFile.fileType;
            requestBody.extension = uploadedFile.extension;
        }

        const response = await fetch(`${BACKEND_URL}/execute`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-ID': SESSION_ID // Session isolation
            },
            body: JSON.stringify(requestBody),
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
            cell.lastOutput = `ERROR: ${errorText}`; // Also save to lastOutput for context

            // Mark that this cell had an error (for auto-run to detect)
            cell.hadError = true;

            // Track execution metadata
            cell.executed = true;
            cell.executionTime = Date.now();
            lastErrorCellId = cellId; // Track globally for smart context detection
            lastExecutedCellId = cellId;

            // Create error display with AI fix button
            outputDiv.innerHTML = `
                <div class="error-text-selectable" data-cell-id="${cellId}" style="margin-bottom: 8px; user-select: text; cursor: text;">${escapeHtml(errorText)}</div>
                <button onclick="askAIToFix('${cellId}')" style="display: inline-block; background: transparent; color: #569cd6; border: 1px solid #569cd6; padding: 3px 8px; border-radius: 2px; cursor: pointer; font-size: 13px; font-weight: 500; line-height: 1.2; white-space: nowrap; text-indent: 0; margin: 0;">Ask AI to Fix</button>
            `;

            // Make error text selectable with "Explain" button
            const errorTextDiv = outputDiv.querySelector('.error-text-selectable');
            if (errorTextDiv) {
                errorTextDiv.addEventListener('mouseup', function() {
                    const selection = window.getSelection();
                    const selectedText = selection.toString().trim();

                    if (selectedText.length > 0) {
                        // Store selection
                        selectedCode = selectedText;
                        selectedCellId = cellId;
                        selectionContext = 'error';

                        // Get position of selection end
                        const range = selection.getRangeAt(0);
                        const rect = range.getBoundingClientRect();

                        // Show explain button (fixed position, so no need for scrollY)
                        const explainBtn = document.getElementById('explainBtn');
                        if (explainBtn) {
                            explainBtn.style.left = (rect.right + 10) + 'px';
                            explainBtn.style.top = (rect.bottom + 5) + 'px';
                            explainBtn.style.display = 'block';
                        }
                    }
                });
            }
        } else {
            // Success! Build output HTML
            cell.hadError = false; // Mark as successful

            // Track execution metadata
            cell.executed = true;
            cell.executionTime = Date.now();
            lastExecutedCellId = cellId;

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
                // Save output to cell for AI context awareness
                cell.lastOutput = outputDiv.textContent || output;
                // Add context-aware suggestion buttons
                addSuggestionButtons(cellId, output);
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

        // Don't auto-scroll - let user stay at current position
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

// File upload handler - CLIENT-SIDE ONLY (no server upload)
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Check file size (max 10MB for browser memory safety)
    const MAX_SIZE = 10 * 1024 * 1024; // 10MB
    if (file.size > MAX_SIZE) {
        addAssistantMessage(`File too large: ${(file.size / 1024 / 1024).toFixed(2)} MB. Maximum is 10 MB.\n\nPlease use a smaller file or filter your data first.`);
        event.target.value = '';
        return;
    }

    // Show loading message
    addAssistantMessage(`Loading ${file.name} into browser memory...`);

    try {
        // Read file into browser memory using FileReader API
        const fileData = await readFileAsBase64(file);

        // Get file extension
        const extension = file.name.split('.').pop().toLowerCase();

        // Validate file type
        const validExtensions = ['csv', 'xlsx', 'xls', 'json', 'txt'];
        if (!validExtensions.includes(extension)) {
            addAssistantMessage(`Unsupported file type: .${extension}\n\nSupported types: CSV, Excel (xlsx/xls), JSON, TXT`);
            event.target.value = '';
            return;
        }

        // Store file in browser memory
        uploadedFile = {
            filename: file.name,
            fileData: fileData, // base64 encoded
            fileType: file.type,
            fileSize: file.size,
            extension: extension
        };

        // Update file indicator UI
        updateFileIndicator();

        // Show success message
        const sizeMB = (file.size / 1024 / 1024).toFixed(2);
        addAssistantMessage(`File loaded: ${file.name} (${sizeMB} MB) - Stored in browser\n\nI can now generate code to analyze this file. Just ask me!`);

        // Reset file input
        event.target.value = '';

    } catch (error) {
        addAssistantMessage(`File load error: ${error.message}`);
        event.target.value = '';
    }
}

// Helper: Read file as base64 using FileReader API
function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            // Extract base64 data (remove "data:...;base64," prefix)
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsDataURL(file);
    });
}

// Update file indicator UI
function updateFileIndicator() {
    const indicator = document.getElementById('fileIndicator');
    const nameElement = document.getElementById('fileIndicatorName');
    const sizeElement = document.getElementById('fileIndicatorSize');

    if (uploadedFile) {
        // Update indicator content
        nameElement.textContent = uploadedFile.filename;
        const sizeMB = (uploadedFile.fileSize / 1024 / 1024).toFixed(2);
        sizeElement.textContent = `(${sizeMB} MB)`;

        // Show indicator
        indicator.style.display = 'block';

        // Adjust content padding to account for indicator
        contentContainer.style.paddingTop = '170px';
    } else {
        // Hide indicator
        indicator.style.display = 'none';

        // Reset content padding
        contentContainer.style.paddingTop = '130px';
    }
}

// Clear uploaded file from browser memory
function clearUploadedFile() {
    if (!uploadedFile) return;

    // Clear the file from memory
    uploadedFile = null;

    // Update UI
    updateFileIndicator();

    // Show message
    addAssistantMessage('File removed from browser memory.');
}

// Code selection "Explain" button functions
function showExplainButton(editor, cellId, code) {
    const explainBtn = document.getElementById('explainBtn');
    selectedCode = code;
    selectedCellId = cellId;
    selectionContext = 'code';

    // Get selection end position (where user stopped selecting)
    const selection = editor.listSelections()[0];
    const endPos = selection.to();
    const coords = editor.charCoords(endPos, 'page');

    // Position button below and slightly to the right of selection end
    explainBtn.style.left = (coords.left + 10) + 'px';
    explainBtn.style.top = (coords.bottom + 5) + 'px';
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
    } else if (context === 'error') {
        // Explaining error message
        let contextCode = '';
        if (cellId && cells[cellId] && cells[cellId].editor) {
            contextCode = cells[cellId].editor.getValue();
        }

        if (contextCode) {
            userMessage = `Explain this error:\n\n${textToExplain}\n\nFrom code:\n\`\`\`python\n${contextCode}\n\`\`\``;
            aiMessage = `Explain this Python error in simple terms:\n\nError: ${textToExplain}\n\nFrom code:\n${contextCode}\n\nWhat does this error mean and how can I fix it?`;
        } else {
            userMessage = `Explain this error:\n\n${textToExplain}`;
            aiMessage = `Explain this Python error in simple terms:\n\n${textToExplain}\n\nWhat does this error mean?`;
        }
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

    // Set insert position: if explaining code/error/output from a cell, insert response after that cell
    if (cellId) {
        insertAfterCellId = cellId;
    }

    // Add user message
    addUserMessage(userMessage);

    // Show loading
    const loadingId = addLoading();
    sendBtn.disabled = true;

    try {
        const response = await fetch(`${BACKEND_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-ID': SESSION_ID  // Add session ID for context
            },
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

// Run all cells sequentially (wait for each to finish before starting next)
async function runAllCellsSequentially() {
    // Get all cell IDs in DOM order (top to bottom)
    const codeArea = document.getElementById('codeArea');
    const cellDivs = Array.from(codeArea.querySelectorAll('[id^="cell-"]'));
    const cellIds = cellDivs.map(div => div.id);

    if (cellIds.length === 0) {
        addAssistantMessage('No cells to run');
        return;
    }

    addAssistantMessage(`Running ${cellIds.length} cells sequentially...`);

    // Run each cell and wait for completion
    for (const cellId of cellIds) {
        const cell = cells[cellId];
        if (!cell || !cell.editor) continue;

        // Run cell and wait for it to complete
        await runCell(cellId);

        // Small delay between cells
        await new Promise(resolve => setTimeout(resolve, 100));
    }

    addAssistantMessage('All cells executed');
}

// Restart kernel and run all cells
async function restartAndRunAll() {
    if (!confirm('Restart kernel and run all cells?\n\nThis will clear all variables and execute all cells from top to bottom.')) {
        return;
    }

    try {
        // First restart the kernel
        const response = await fetch(`${BACKEND_URL}/restart_kernel`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-ID': SESSION_ID  // Required for session isolation
            }
        });

        const data = await response.json();

        if (data.status === 'ok') {
            addAssistantMessage('Kernel restarted - running all cells...');

            // Wait a moment for kernel to be ready
            await new Promise(resolve => setTimeout(resolve, 500));

            // Run all cells sequentially
            await runAllCellsSequentially();
        } else {
            // Handle error responses properly
            const errorMsg = data.message || data.error || 'Unknown error';
            addAssistantMessage(`Kernel restart failed: ${errorMsg}`);
        }
    } catch (error) {
        addAssistantMessage(`Restart & Run All error: ${error.message}`);
    }
}

// Add context-aware suggestion buttons after cell output
function addSuggestionButtons(cellId, outputContent) {
    const cell = cells[cellId];
    if (!cell || !cell.outputDiv) return;

    // Remove existing suggestion buttons if any
    const existingSuggestions = cell.outputDiv.querySelector('.suggestion-buttons');
    if (existingSuggestions) {
        existingSuggestions.remove();
    }

    // Analyze output to determine context
    const outputText = cell.outputDiv.textContent || '';
    const outputHTML = outputContent || '';

    // Detect output type
    const hasTable = outputHTML.includes('<table') || outputHTML.includes('DataFrame');
    const hasPlot = outputHTML.includes('plotly-graph-div') || outputText.includes('matplotlib plot');
    const hasNumericResult = /^[\d\.\,\-\+\s]+$/.test(outputText.trim()) && outputText.trim().length < 100;
    const hasDataQualityReport = outputText.includes('DATA QUALITY REPORT') || outputText.includes('📊');
    const hasMissingValues = outputText.includes('Missing Values') || outputText.includes('NaN');

    // Generate context-aware suggestions
    let suggestions = [];

    if (hasDataQualityReport) {
        suggestions = [
            { text: 'Visualize distribution', prompt: 'Create visualizations of the most important columns' },
            { text: 'Handle missing values', prompt: 'How can I best handle the missing values?' },
            { text: 'Show correlations', prompt: 'Create a correlation heatmap' }
        ];
    } else if (hasTable) {
        suggestions = [
            { text: 'Explain output', prompt: 'Explain this output' },
            { text: 'Visualize data', prompt: 'Visualize this data' },
            { text: 'Show statistics', prompt: 'Show statistics for this data' }
        ];
    } else if (hasPlot) {
        suggestions = [
            { text: 'Improve styling', prompt: 'Improve the styling of this visualization' },
            { text: 'Add interactivity', prompt: 'Make this visualization more interactive' },
            { text: 'Export plot', prompt: 'How can I export this plot?' }
        ];
    } else if (hasNumericResult) {
        suggestions = [
            { text: 'Visualize result', prompt: 'Visualize this result' },
            { text: 'Explain calculation', prompt: 'Explain how this was calculated' },
            { text: 'Compare values', prompt: 'Compare these values with statistics' }
        ];
    } else if (hasMissingValues) {
        suggestions = [
            { text: 'Fill missing values', prompt: 'Fill in the missing values' },
            { text: 'Remove nulls', prompt: 'Remove rows with missing values' },
            { text: 'Analyze pattern', prompt: 'Analyze the pattern of missing values' }
        ];
    } else {
        // Default suggestions
        suggestions = [
            { text: 'Explain output', prompt: 'Explain this output' },
            { text: 'Next step', prompt: 'What is the next step?' },
            { text: 'Visualize', prompt: 'Can you visualize this?' }
        ];
    }

    // Create suggestion buttons container
    const suggestionsDiv = document.createElement('div');
    suggestionsDiv.className = 'suggestion-buttons';
    suggestionsDiv.style.cssText = `
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #3c3c3c;
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    `;

    // Add each suggestion button
    suggestions.forEach(suggestion => {
        const btn = document.createElement('button');
        btn.textContent = suggestion.text;
        btn.style.cssText = `
            background: transparent;
            color: #4ec9b0;
            border: 1px solid #4ec9b0;
            padding: 3px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s;
        `;

        // Hover effect
        btn.onmouseenter = () => {
            btn.style.background = '#4ec9b0';
            btn.style.color = '#1e1e1e';
        };
        btn.onmouseleave = () => {
            btn.style.background = 'transparent';
            btn.style.color = '#4ec9b0';
        };

        // Click handler - send directly to AI
        btn.onclick = () => {
            // Set context to insert after this cell
            insertAfterCellId = cellId;
            // Set the prompt
            messageInput.value = suggestion.prompt;
            // Send immediately
            sendMessage();
        };

        suggestionsDiv.appendChild(btn);
    });

    // Append to output div
    cell.outputDiv.appendChild(suggestionsDiv);
}

// Ask about this cell - sets context for smart insertion
function askAboutCell(cellId) {
    const cell = cells[cellId];
    if (!cell || !cell.editor) return;

    // Set context: responses will appear after this cell
    insertAfterCellId = cellId;

    // Focus chat input and scroll it into view
    messageInput.focus();
    messageInput.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // Add a visual hint
    const code = cell.editor.getValue().trim();
    const cellNumber = Object.keys(cells).indexOf(cellId) + 1;

    // If cell has error, suggest asking about it
    if (cell.hadError) {
        messageInput.value = `Waarom geeft deze cel een error?`;
    } else if (cell.executed && cell.lastOutput) {
        messageInput.value = `Leg deze output uit`;
    } else if (code) {
        messageInput.value = `Wat doet deze code?`;
    } else {
        messageInput.value = ``;
    }

    messageInput.select();

    console.log(`Context set: responses will appear after cell ${cellNumber}`);
}

// Ask AI to fix an error
async function askAIToFix(cellId) {
    const cell = cells[cellId];
    if (!cell || !cell.editor || !cell.lastError) return;

    const code = cell.editor.getValue();
    const errorMessage = cell.lastError;

    // Set insert position: new cell should appear right after this cell
    insertAfterCellId = cellId;

    // Show loading animation
    const loadingId = addLoading();

    try {
        // Get recent cells for context (last 3 cells)
        const cellsArray = Object.values(cells);
        const recentCells = cellsArray
            .slice(Math.max(0, cellsArray.length - 3))
            .map(c => ({
                type: 'code',
                code: c.editor ? c.editor.getValue() : '',
                output: c.lastOutput || ''
            }));

        // Call chat API with special fix prompt
        const response = await fetch(`${BACKEND_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-ID': SESSION_ID
            },
            body: JSON.stringify({
                message: `ERROR FIX REQUEST - Generate fixed code:\n\nBroken code:\n${code}\n\nError:\n${errorMessage}\n\nInstructions:\n1. Respond with CODE: containing the FIXED version\n2. Fix the error but keep the same functionality\n3. Only fix what's broken, don't add features`,
                history: [], // No chat history needed for fixes
                recent_cells: recentCells,
                uploaded_file: uploadedFile ? {
                    filename: uploadedFile.filename,
                    extension: uploadedFile.extension,
                    fileSize: uploadedFile.fileSize
                } : null
            })
        });

        const data = await response.json();

        // Remove loading animation
        removeLoading(loadingId);

        if (data.error) {
            addAssistantMessage(`Failed to generate fix: ${data.error}`, 'text');
            return;
        }

        // Show explanation in chat
        const errorType = errorMessage.includes('Syntax') ? 'Syntax error' :
                         errorMessage.includes('Validation') ? 'Validation error' :
                         'Runtime error';
        const explanation = `**Fixed!** The error was a **${errorType}**.\n\nI've created a corrected version in a new cell below.`;
        addAssistantMessage(explanation, 'text');

        // Create new cell with fixed code
        if (data.type === 'code' && data.message) {
            // Use the existing addCodeCell function to create the new cell
            const newCellId = addCodeCell(data.message);

            // Clear the insertAfter setting
            insertAfterCellId = null;

            // Scroll to the newly created cell
            const newCellDiv = document.getElementById(newCellId);
            if (newCellDiv) {
                newCellDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        } else {
            // Fallback: just show the response in chat
            addAssistantMessage(data.message, data.type || 'text');
            // Clear the insertAfter setting
            insertAfterCellId = null;
        }

    } catch (error) {
        console.error('AI Fix Error:', error);
        // Remove loading animation on error
        removeLoading(loadingId);
        addAssistantMessage(`Error communicating with AI: ${error.message}`, 'text');
        // Clear the insertAfter setting
        insertAfterCellId = null;
    }
}
