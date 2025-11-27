// Cytherea Console JavaScript

let pyodide = null;
let cythereaInterface = null;
let visualizationActive = true;
let mindCanvas = null;
let ctx = null;
let animationId = null;

// Color palette
const COLORS = {
    soft_gold: "#EED88F",
    moon_shell: "#F4F1EB",
    deep_tide: "#234C67",
    rose_quartz: "#E5A4C5",
    midnight_petal: "#1C1A27"
};

window.initializeCythereaConsole = async function() {
    console.log("Initializing Cytherea Console...");
    
    // Initialize canvas
    mindCanvas = document.getElementById('mind-canvas');
    if (mindCanvas) {
        ctx = mindCanvas.getContext('2d');
        resizeCanvas();
    }
    
    // Load Pyodide
    pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
    });
    
    // Load required packages
    await pyodide.loadPackage(['numpy']);
    
    // Load Cytherea Python code
    const pythonCode = await fetch('../site/interactive/py/child_assistant_pyodide.py').then(r => r.text());
    await pyodide.runPython(pythonCode);
    
    // Initialize Cytherea
    await pyodide.runPythonAsync(`
# Create Cytherea instance
cytherea = CythereaWeb()
await cytherea.initialize()

# Get initial greeting
greeting = cytherea.get_greeting()
    `);
    
    // Set up event handlers
    setupEventHandlers();
    
    // Start visualization
    if (visualizationActive && ctx) {
        startVisualization();
    }
    
    // Display initial greeting
    const greeting = pyodide.globals.get('greeting');
    addChatMessage('Cytherea', greeting, 'cytherea');
    
    console.log("Cytherea Console initialized!");
}

function setupEventHandlers() {
    // Send button
    const sendBtn = document.getElementById('send-button');
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }
    
    // Enter key in textarea
    const input = document.getElementById('user-input');
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    
    // Visualization toggle
    const toggleBtn = document.getElementById('toggle-viz');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            visualizationActive = !visualizationActive;
            if (visualizationActive) {
                startVisualization();
            } else {
                stopVisualization();
            }
        });
    }
    
    // View internals
    const internalsBtn = document.getElementById('view-internals');
    if (internalsBtn) {
        internalsBtn.addEventListener('click', showInternalState);
    }
    
    // Inspector close
    const closeBtn = document.getElementById('close-inspector');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            const inspector = document.getElementById('state-inspector');
            if (inspector) inspector.classList.add('hidden');
        });
    }
    
    // Window resize
    window.addEventListener('resize', resizeCanvas);
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addChatMessage('You', message, 'user');
    
    // Clear input
    input.value = '';
    
    // Process message through Cytherea
    try {
        const response = await pyodide.runPythonAsync(`
import json
response = await cytherea.process_input("${message.replace(/"/g, '\\"')}")
json.dumps(response)
        `);
        
        const responseData = JSON.parse(response);
        
        // Add Cytherea's response
        addChatMessage('Cytherea', responseData.text, 'cytherea');
        
        // Update status displays
        updateStatus(responseData);
        
    } catch (error) {
        console.error("Error processing message:", error);
        addChatMessage('System', 'Error processing message', 'error');
    }
}

function addChatMessage(sender, message, className) {
    const chatHistory = document.getElementById('chat-history');
    if (!chatHistory) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;
    
    const senderSpan = document.createElement('span');
    senderSpan.className = 'sender';
    senderSpan.textContent = sender + ': ';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    contentDiv.innerHTML = formatMessage(message);
    
    messageDiv.appendChild(senderSpan);
    messageDiv.appendChild(contentDiv);
    chatHistory.appendChild(messageDiv);
    
    // Scroll to bottom
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function formatMessage(message) {
    // Convert *action* to italics
    message = message.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Convert line breaks
    message = message.replace(/\n/g, '<br>');
    return message;
}

function updateStatus(response) {
    // Update mood
    const moodDisplay = document.getElementById('mood-display');
    if (moodDisplay && response.mood) {
        moodDisplay.textContent = response.mood;
    }
}

function resizeCanvas() {
    if (!mindCanvas) return;
    const container = document.getElementById('cytherea-visualization');
    if (container) {
        mindCanvas.width = container.clientWidth - 20;
        mindCanvas.height = 200;
    }
}

function startVisualization() {
    if (animationId || !ctx) return;
    
    function animate() {
        drawMindState();
        animationId = requestAnimationFrame(animate);
    }
    
    animate();
}

function stopVisualization() {
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
}

function drawMindState() {
    if (!ctx || !mindCanvas) return;
    
    // Clear canvas
    ctx.fillStyle = COLORS.midnight_petal;
    ctx.fillRect(0, 0, mindCanvas.width, mindCanvas.height);
    
    // Draw simple animated pattern
    const time = Date.now() / 1000;
    
    // Draw coherence waves
    ctx.strokeStyle = COLORS.rose_quartz;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.6;
    
    ctx.beginPath();
    for (let x = 0; x < mindCanvas.width; x++) {
        const y = mindCanvas.height / 2 + Math.sin(x * 0.02 + time) * 30;
        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
    
    // Draw consciousness particles
    ctx.fillStyle = COLORS.soft_gold;
    ctx.globalAlpha = 0.8;
    
    for (let i = 0; i < 5; i++) {
        const x = (i + 0.5) * (mindCanvas.width / 5);
        const y = mindCanvas.height / 2 + Math.sin(time + i) * 20;
        const radius = 5 + Math.sin(time * 2 + i) * 2;
        
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }
    
    ctx.globalAlpha = 1.0;
}

async function showInternalState() {
    try {
        const stateJson = await pyodide.runPythonAsync(`
import json
introspection = await cytherea.get_introspection()
json.dumps(introspection, indent=2)
        `);
        
        const stateDisplay = document.getElementById('state-json');
        if (stateDisplay) {
            stateDisplay.textContent = stateJson;
        }
        
        const inspector = document.getElementById('state-inspector');
        if (inspector) {
            inspector.classList.remove('hidden');
        }
        
    } catch (error) {
        console.error("Error getting internal state:", error);
    }
}

// Add custom styles for Cytherea
const style = document.createElement('style');
style.textContent = `
.cytherea-console-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

#cytherea-header {
    text-align: center;
    margin-bottom: 30px;
}

.cytherea-glyph {
    width: 100px;
    height: 100px;
    margin-bottom: 10px;
}

.status-panel {
    display: flex;
    justify-content: space-around;
    background: ${COLORS.moon_shell};
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.viz-container {
    background: ${COLORS.midnight_petal};
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 20px;
    position: relative;
}

.chat-panel {
    background: white;
    border: 1px solid ${COLORS.moon_shell};
    border-radius: 10px;
    padding: 20px;
}

#chat-history {
    height: 400px;
    overflow-y: auto;
    margin-bottom: 20px;
    padding: 10px;
    background: ${COLORS.moon_shell}40;
    border-radius: 5px;
}

.message {
    margin-bottom: 15px;
    padding: 10px;
    border-radius: 5px;
}

.message.user {
    background: ${COLORS.deep_tide}20;
    margin-left: 50px;
}

.message.cytherea {
    background: ${COLORS.rose_quartz}20;
    margin-right: 50px;
}

.message .sender {
    font-weight: bold;
    display: block;
    margin-bottom: 5px;
}

#input-container {
    display: flex;
    gap: 10px;
}

#user-input {
    flex: 1;
    padding: 10px;
    border: 1px solid ${COLORS.moon_shell};
    border-radius: 5px;
    resize: none;
    font-family: inherit;
}

#send-button {
    background: ${COLORS.soft_gold};
    color: ${COLORS.midnight_petal};
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
}

.inspector-panel {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: white;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 10px 50px rgba(0,0,0,0.3);
    max-width: 800px;
    max-height: 80vh;
    overflow-y: auto;
    z-index: 1000;
}

.hidden {
    display: none;
}
`;
document.head.appendChild(style);