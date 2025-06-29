const { GoogleGenAI } = require('@google/genai');
const { BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const { saveDebugAudio } = require('../audioUtils');
const { getSystemPrompt } = require('./prompts');

// Conversation tracking variables
let currentSessionId = null;
let currentTranscription = '';
let conversationHistory = [];
let isInitializingSession = false;

// Audio capture variables
let systemAudioProc = null;
let messageBuffer = '';

// Session management variables
let sessionRetryCount = 0;
const MAX_RETRY_ATTEMPTS = 3;
let isSessionClosing = false;
let connectionHealthTimer = null;

// Token monitoring
let totalTokensUsed = 0;
const TOKEN_LIMIT_WARNING = 15000;
const TOKEN_LIMIT_RESTART = 18000;

function sendToRenderer(channel, data) {
    const windows = BrowserWindow.getAllWindows();
    if (windows.length > 0) {
        windows[0].webContents.send(channel, data);
    }
}

// Conversation management functions
function initializeNewSession() {
    currentSessionId = Date.now().toString();
    currentTranscription = '';
    conversationHistory = [];
    totalTokensUsed = 0; // Reset token counter
    console.log('New conversation session started:', currentSessionId);
}

function saveConversationTurn(transcription, aiResponse) {
    if (!currentSessionId) {
        initializeNewSession();
    }

    const conversationTurn = {
        timestamp: Date.now(),
        transcription: transcription.trim(),
        ai_response: aiResponse.trim(),
    };

    conversationHistory.push(conversationTurn);
    console.log('Saved conversation turn:', conversationTurn);

    // Send to renderer to save in IndexedDB
    sendToRenderer('save-conversation-turn', {
        sessionId: currentSessionId,
        turn: conversationTurn,
        fullHistory: conversationHistory,
    });
}

function getCurrentSessionData() {
    return {
        sessionId: currentSessionId,
        history: conversationHistory,
    };
}

async function getEnabledTools() {
    const tools = [];

    // Check if Google Search is enabled (default: true)
    const googleSearchEnabled = await getStoredSetting('googleSearchEnabled', 'true');
    console.log('Google Search enabled:', googleSearchEnabled);

    if (googleSearchEnabled === 'true') {
        tools.push({ googleSearch: {} });
        console.log('Added Google Search tool');
    } else {
        console.log('Google Search tool disabled');
    }

    return tools;
}

async function getStoredSetting(key, defaultValue) {
    try {
        const windows = BrowserWindow.getAllWindows();
        if (windows.length > 0) {
            // Wait a bit for the renderer to be ready
            await new Promise(resolve => setTimeout(resolve, 100));

            // Try to get setting from renderer process localStorage
            const value = await windows[0].webContents.executeJavaScript(`
                (function() {
                    try {
                        if (typeof localStorage === 'undefined') {
                            console.log('localStorage not available yet for ${key}');
                            return '${defaultValue}';
                        }
                        const stored = localStorage.getItem('${key}');
                        console.log('Retrieved setting ${key}:', stored);
                        return stored || '${defaultValue}';
                    } catch (e) {
                        console.error('Error accessing localStorage for ${key}:', e);
                        return '${defaultValue}';
                    }
                })()
            `);
            return value;
        }
    } catch (error) {
        console.error('Error getting stored setting for', key, ':', error.message);
    }
    console.log('Using default value for', key, ':', defaultValue);
    return defaultValue;
}

async function initializeGeminiSession(apiKey, customPrompt = '', profile = 'interview', language = 'en-US', geminiSessionRef = null) {
    if (isInitializingSession) {
        console.log('Session initialization already in progress');
        return false;
    }

    isInitializingSession = true;
    isSessionClosing = false;
    sendToRenderer('session-initializing', true);

    const client = new GoogleGenAI({
        vertexai: false,
        apiKey: apiKey,
    });

    // Get enabled tools first to determine Google Search status
    const enabledTools = await getEnabledTools();
    const googleSearchEnabled = enabledTools.some(tool => tool.googleSearch);

    const systemPrompt = getSystemPrompt(profile, customPrompt, googleSearchEnabled);

    // Initialize new conversation session
    initializeNewSession();

    try {
        const session = await client.live.connect({
            model: 'gemini-live-2.5-flash-preview',
            callbacks: {
                onopen: () => {
                    console.log('Gemini session opened successfully');
                    sessionRetryCount = 0; // Reset retry count on successful connection
                    sendToRenderer('update-status', 'Connected to Gemini - Starting recording...');
                    
                    // Start connection health monitoring
                    if (geminiSessionRef) {
                        startConnectionHealthMonitoring(geminiSessionRef);
                    }
                },
                onmessage: (message) => {
                    console.log('----------------', message);

                    // Handle transcription input
                    if (message.serverContent?.inputTranscription?.text) {
                        currentTranscription += message.serverContent.inputTranscription.text;
                    }

                    // Handle AI model response
                    if (message.serverContent?.modelTurn?.parts) {
                        for (const part of message.serverContent.modelTurn.parts) {
                            console.log(part);
                            if (part.text) {
                                messageBuffer += part.text;
                            }
                        }
                    }

                    if (message.serverContent?.generationComplete) {
                        sendToRenderer('update-response', messageBuffer);

                        // Save conversation turn when we have both transcription and AI response
                        if (currentTranscription && messageBuffer) {
                            saveConversationTurn(currentTranscription, messageBuffer);
                            currentTranscription = ''; // Reset for next turn
                        }

                        messageBuffer = '';
                    }

                    if (message.serverContent?.turnComplete) {
                        sendToRenderer('update-status', 'Listening...');
                    }

                    // Log usage metadata for monitoring
                    if (message.usageMetadata) {
                        console.log('Token usage:', message.usageMetadata);
                        totalTokensUsed = message.usageMetadata.totalTokenCount || 0;
                        
                        // Check for high token usage and warn
                        if (totalTokensUsed > TOKEN_LIMIT_WARNING) {
                            console.warn('⚠️ High token usage detected:', totalTokensUsed);
                            sendToRenderer('update-status', `High token usage (${totalTokensUsed}) - consider restarting session`);
                        }
                        
                        // Auto-restart session if approaching limits
                        if (totalTokensUsed > TOKEN_LIMIT_RESTART) {
                            console.warn('🔄 Token limit approaching, restarting session automatically');
                            sendToRenderer('update-status', 'Restarting session due to token limit...');
                            
                            // Schedule restart after a brief delay
                            // Note: We need to get the geminiSessionRef from the calling context
                            setTimeout(async () => {
                                // This will be handled by the IPC handler which has access to geminiSessionRef
                                sendToRenderer('auto-restart-session', { apiKey, customPrompt, profile, language });
                            }, 1000);
                        }
                    }
                },
                onerror: (e) => {
                    console.error('Gemini session error:', e);
                    sendToRenderer('update-status', 'Error: ' + e.message);
                    
                    // Attempt to recover from certain types of errors
                    if (!isSessionClosing && shouldRetryConnection(e)) {
                        handleConnectionError(apiKey, customPrompt, profile, language, geminiSessionRef);
                    }
                },
                onclose: (e) => {
                    console.log('Session closed:', e.reason);
                    stopConnectionHealthMonitoring();
                    
                    if (!isSessionClosing) {
                        sendToRenderer('update-status', 'Session closed unexpectedly');
                        
                        // Attempt to reconnect if not intentionally closed
                        if (shouldRetryConnection(e)) {
                            handleConnectionError(apiKey, customPrompt, profile, language, geminiSessionRef);
                        }
                    } else {
                        sendToRenderer('update-status', 'Session closed');
                    }
                },
            },
            config: {
                responseModalities: ['TEXT'],
                tools: enabledTools,
                inputAudioTranscription: {},
                contextWindowCompression: { 
                    slidingWindow: {
                        // Reduce context window to prevent token overflow
                        maxTokens: 10000
                    } 
                },
                speechConfig: { languageCode: language },
                systemInstruction: {
                    parts: [{ text: systemPrompt }],
                },
            },
        });

        isInitializingSession = false;
        sendToRenderer('session-initializing', false);
        return session;
    } catch (error) {
        console.error('Failed to initialize Gemini session:', error);
        isInitializingSession = false;
        sendToRenderer('session-initializing', false);
        
        // Attempt retry for certain errors
        if (!isSessionClosing && shouldRetryConnection(error)) {
            return handleConnectionError(apiKey, customPrompt, profile, language, geminiSessionRef);
        }
        
        return null;
    }
}

function shouldRetryConnection(error) {
    if (sessionRetryCount >= MAX_RETRY_ATTEMPTS) {
        console.log('Max retry attempts reached, not retrying');
        return false;
    }
    
    // Retry for network errors, cancellation, or timeout issues
    const retryableErrors = [
        'CANCELLED',
        'UNAVAILABLE', 
        'DEADLINE_EXCEEDED',
        'INTERNAL',
        'network',
        'timeout',
        'connection'
    ];
    
    const errorMessage = error?.message || error?.reason || String(error);
    return retryableErrors.some(pattern => 
        errorMessage.toLowerCase().includes(pattern.toLowerCase())
    );
}

async function handleConnectionError(apiKey, customPrompt, profile, language, geminiSessionRef = null) {
    sessionRetryCount++;
    const delay = Math.min(1000 * (2 ** (sessionRetryCount - 1)), 10000); // Exponential backoff, max 10s
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${sessionRetryCount}/${MAX_RETRY_ATTEMPTS})`);
    sendToRenderer('update-status', `Reconnecting... (${sessionRetryCount}/${MAX_RETRY_ATTEMPTS})`);
    
    setTimeout(async () => {
        console.log('Retrying Gemini session initialization...');
        const session = await initializeGeminiSession(apiKey, customPrompt, profile, language, geminiSessionRef);
        if (session && geminiSessionRef) {
            geminiSessionRef.current = session;
        }
    }, delay);
}

function startConnectionHealthMonitoring(geminiSessionRef) {
    // Clear any existing timer
    stopConnectionHealthMonitoring();
    
    // Check connection health every 30 seconds
    connectionHealthTimer = setInterval(() => {
        // Send a small keep-alive message to check connection
        if (geminiSessionRef?.current && !isSessionClosing) {
            try {
                // Send empty text to test connection
                geminiSessionRef.current.sendRealtimeInput({ text: '' }).catch(error => {
                    console.warn('Connection health check failed:', error);
                });
            } catch (error) {
                console.warn('Connection health check error:', error);
            }
        }
    }, 30000);
}

function stopConnectionHealthMonitoring() {
    if (connectionHealthTimer) {
        clearInterval(connectionHealthTimer);
        connectionHealthTimer = null;
    }
}

function killExistingSystemAudioDump() {
    return new Promise(resolve => {
        console.log('Checking for existing SystemAudioDump processes...');

        // Kill any existing SystemAudioDump processes
        const killProc = spawn('pkill', ['-f', 'SystemAudioDump'], {
            stdio: 'ignore',
        });

        killProc.on('close', code => {
            if (code === 0) {
                console.log('Killed existing SystemAudioDump processes');
            } else {
                console.log('No existing SystemAudioDump processes found');
            }
            resolve();
        });

        killProc.on('error', err => {
            console.log('Error checking for existing processes (this is normal):', err.message);
            resolve();
        });

        // Timeout after 2 seconds
        setTimeout(() => {
            killProc.kill();
            resolve();
        }, 2000);
    });
}

async function startMacOSAudioCapture(geminiSessionRef) {
    if (process.platform !== 'darwin') return false;

    // Kill any existing SystemAudioDump processes first
    await killExistingSystemAudioDump();

    console.log('Starting macOS audio capture with SystemAudioDump...');

    const { app } = require('electron');
    const path = require('path');

    let systemAudioPath;
    if (app.isPackaged) {
        systemAudioPath = path.join(process.resourcesPath, 'SystemAudioDump');
    } else {
        systemAudioPath = path.join(__dirname, '../assets', 'SystemAudioDump');
    }

    console.log('SystemAudioDump path:', systemAudioPath);

    systemAudioProc = spawn(systemAudioPath, [], {
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    if (!systemAudioProc.pid) {
        console.error('Failed to start SystemAudioDump');
        return false;
    }

    console.log('SystemAudioDump started with PID:', systemAudioProc.pid);

    const CHUNK_DURATION = 0.1;
    const SAMPLE_RATE = 24000;
    const BYTES_PER_SAMPLE = 2;
    const CHANNELS = 2;
    const CHUNK_SIZE = SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * CHUNK_DURATION;

    let audioBuffer = Buffer.alloc(0);

    systemAudioProc.stdout.on('data', data => {
        audioBuffer = Buffer.concat([audioBuffer, data]);

        while (audioBuffer.length >= CHUNK_SIZE) {
            const chunk = audioBuffer.slice(0, CHUNK_SIZE);
            audioBuffer = audioBuffer.slice(CHUNK_SIZE);

            const monoChunk = CHANNELS === 2 ? convertStereoToMono(chunk) : chunk;
            const base64Data = monoChunk.toString('base64');
            sendAudioToGemini(base64Data, geminiSessionRef);

            if (process.env.DEBUG_AUDIO) {
                console.log(`Processed audio chunk: ${chunk.length} bytes`);
                saveDebugAudio(monoChunk, 'system_audio');
            }
        }

        const maxBufferSize = SAMPLE_RATE * BYTES_PER_SAMPLE * 1;
        if (audioBuffer.length > maxBufferSize) {
            audioBuffer = audioBuffer.slice(-maxBufferSize);
        }
    });

    systemAudioProc.stderr.on('data', data => {
        console.error('SystemAudioDump stderr:', data.toString());
    });

    systemAudioProc.on('close', code => {
        console.log('SystemAudioDump process closed with code:', code);
        systemAudioProc = null;
    });

    systemAudioProc.on('error', err => {
        console.error('SystemAudioDump process error:', err);
        systemAudioProc = null;
    });

    return true;
}

function convertStereoToMono(stereoBuffer) {
    const samples = stereoBuffer.length / 4;
    const monoBuffer = Buffer.alloc(samples * 2);

    for (let i = 0; i < samples; i++) {
        const leftSample = stereoBuffer.readInt16LE(i * 4);
        monoBuffer.writeInt16LE(leftSample, i * 2);
    }

    return monoBuffer;
}

function stopMacOSAudioCapture() {
    if (systemAudioProc) {
        console.log('Stopping SystemAudioDump...');
        systemAudioProc.kill('SIGTERM');
        systemAudioProc = null;
    }
}

async function sendAudioToGemini(base64Data, geminiSessionRef) {
    if (!geminiSessionRef.current || isSessionClosing) return;

    try {
        process.stdout.write('.');
        await geminiSessionRef.current.sendRealtimeInput({
            audio: {
                data: base64Data,
                mimeType: 'audio/pcm;rate=24000',
            },
        });
    } catch (error) {
        console.error('Error sending audio to Gemini:', error);
        
        // If session is broken, stop sending audio
        if (error.message.includes('CANCELLED') || error.message.includes('closed')) {
            console.warn('Session appears to be closed, stopping audio transmission');
        }
    }
}

function setupGeminiIpcHandlers(geminiSessionRef) {
    ipcMain.handle('initialize-gemini', async (event, apiKey, customPrompt, profile = 'interview', language = 'en-US') => {
        const session = await initializeGeminiSession(apiKey, customPrompt, profile, language, geminiSessionRef);
        if (session) {
            geminiSessionRef.current = session;
            return true;
        }
        return false;
    });

    ipcMain.handle('send-audio-content', async (event, { data, mimeType }) => {
        if (!geminiSessionRef.current) return { success: false, error: 'No active Gemini session' };
        try {
            process.stdout.write('.');
            await geminiSessionRef.current.sendRealtimeInput({
                audio: { data: data, mimeType: mimeType },
            });
            return { success: true };
        } catch (error) {
            console.error('Error sending audio:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-image-content', async (event, { data, debug }) => {
        if (!geminiSessionRef.current || isSessionClosing) return { success: false, error: 'No active Gemini session' };

        try {
            if (!data || typeof data !== 'string') {
                console.error('Invalid image data received');
                return { success: false, error: 'Invalid image data' };
            }

            const buffer = Buffer.from(data, 'base64');

            if (buffer.length < 1000) {
                console.error(`Image buffer too small: ${buffer.length} bytes`);
                return { success: false, error: 'Image buffer too small' };
            }

            process.stdout.write('!');
            await geminiSessionRef.current.sendRealtimeInput({
                media: { data: data, mimeType: 'image/jpeg' },
            });

            return { success: true };
        } catch (error) {
            console.error('Error sending image:', error);
            
            // Handle session-related errors
            if (error.message.includes('CANCELLED') || error.message.includes('closed')) {
                console.warn('Session appears to be closed during image send');
                return { success: false, error: 'Session closed' };
            }
            
            // Handle rate limiting
            if (error.message.includes('RESOURCE_EXHAUSTED') || error.message.includes('rate')) {
                console.warn('Rate limit hit during image send');
                return { success: false, error: 'Rate limit exceeded - try reducing image frequency' };
            }
            
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-text-message', async (event, text) => {
        if (!geminiSessionRef.current || isSessionClosing) return { success: false, error: 'No active Gemini session' };

        try {
            if (!text || typeof text !== 'string' || text.trim().length === 0) {
                return { success: false, error: 'Invalid text message' };
            }

            console.log('Sending text message:', text);
            await geminiSessionRef.current.sendRealtimeInput({ text: text.trim() });
            return { success: true };
        } catch (error) {
            console.error('Error sending text:', error);
            
            // Handle session-related errors  
            if (error.message.includes('CANCELLED') || error.message.includes('closed')) {
                console.warn('Session appears to be closed during text send');
                return { success: false, error: 'Session closed' };
            }
            
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('start-macos-audio', async event => {
        if (process.platform !== 'darwin') {
            return {
                success: false,
                error: 'macOS audio capture only available on macOS',
            };
        }

        try {
            const success = await startMacOSAudioCapture(geminiSessionRef);
            return { success };
        } catch (error) {
            console.error('Error starting macOS audio capture:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('stop-macos-audio', async event => {
        try {
            stopMacOSAudioCapture();
            return { success: true };
        } catch (error) {
            console.error('Error stopping macOS audio capture:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('close-session', async event => {
        try {
            isSessionClosing = true;
            stopConnectionHealthMonitoring();
            stopMacOSAudioCapture();

            // Cleanup any pending resources and stop audio/video capture
            if (geminiSessionRef.current) {
                try {
                    await geminiSessionRef.current.close();
                } catch (closeError) {
                    console.warn('Error during session close (this is usually safe to ignore):', closeError.message);
                } finally {
                    geminiSessionRef.current = null;
                }
            }

            // Reset session state
            sessionRetryCount = 0;
            isSessionClosing = false;

            return { success: true };
        } catch (error) {
            console.error('Error closing session:', error);
            isSessionClosing = false;
            return { success: false, error: error.message };
        }
    });

    // Conversation history IPC handlers
    ipcMain.handle('get-current-session', async event => {
        try {
            return { success: true, data: getCurrentSessionData() };
        } catch (error) {
            console.error('Error getting current session:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('start-new-session', async event => {
        try {
            initializeNewSession();
            return { success: true, sessionId: currentSessionId };
        } catch (error) {
            console.error('Error starting new session:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('update-google-search-setting', async (event, enabled) => {
        try {
            console.log('Google Search setting updated to:', enabled);
            // The setting is already saved in localStorage by the renderer
            // This is just for logging/confirmation
            return { success: true };
        } catch (error) {
            console.error('Error updating Google Search setting:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('restart-session-due-to-tokens', async (event, { apiKey, customPrompt, profile, language }) => {
        try {
            const session = await restartSessionDueToTokens(apiKey, customPrompt, profile, language, geminiSessionRef);
            return { success: !!session };
        } catch (error) {
            console.error('Error restarting session due to tokens:', error);
            return { success: false, error: error.message };
        }
    });
}

async function restartSessionDueToTokens(apiKey, customPrompt, profile, language, geminiSessionRef) {
    try {
        console.log('Restarting session due to token limit...');
        
        // Close current session gracefully
        if (geminiSessionRef.current) {
            try {
                await geminiSessionRef.current.close();
            } catch (closeError) {
                console.warn('Error closing session for restart:', closeError.message);
            }
            geminiSessionRef.current = null;
        }
        
        // Reset token counter
        totalTokensUsed = 0;
        
        // Start new session
        const session = await initializeGeminiSession(apiKey, customPrompt, profile, language, geminiSessionRef);
        if (session) {
            // Update the session reference
            geminiSessionRef.current = session;
            sendToRenderer('session-restarted', { reason: 'token-limit' });
            console.log('✅ Session restarted successfully due to token limit');
        } else {
            console.error('❌ Failed to restart session due to token limit');
            sendToRenderer('update-status', 'Failed to restart session - please restart manually');
        }
        
        return session;
    } catch (error) {
        console.error('Error restarting session due to token limit:', error);
        sendToRenderer('update-status', 'Error restarting session - please restart manually');
        return null;
    }
}

module.exports = {
    initializeGeminiSession,
    getEnabledTools,
    getStoredSetting,
    sendToRenderer,
    initializeNewSession,
    saveConversationTurn,
    getCurrentSessionData,
    killExistingSystemAudioDump,
    startMacOSAudioCapture,
    convertStereoToMono,
    stopMacOSAudioCapture,
    sendAudioToGemini,
    setupGeminiIpcHandlers,
    restartSessionDueToTokens,
    shouldRetryConnection,
    handleConnectionError,
};
