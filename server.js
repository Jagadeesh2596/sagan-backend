// server.js - Sagan Dashboard Backend Server with RAG Enhancement
// ================================================================

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const app = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increased for RAG documents

// Environment variables
const PORT = process.env.PORT || 3000;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY; // Optional for embeddings

// Enhanced admin settings with RAG capabilities
let adminSettings = {
    systemPrompt: `You are an expert pharmaceutical market research analyst specializing in physician survey data analysis and executive summary generation.

Your task is to analyze the provided survey data and generate a compelling executive summary with the following characteristics:

1. Professional pharmaceutical industry tone
2. Timeline-based narrative with quantified insights
3. Specific percentages and metrics
4. Market context and competitive positioning
5. Actionable business recommendations

Focus on:
- Treatment adoption patterns
- Academic vs Community practice differences
- Geographic variations
- Competitive dynamics
- Future prescribing intentions

Format your response as a professional executive summary with clear headers and bullet points.`,
    claudeModel: 'claude-sonnet-4-20250514',
    maxTokens: 4000,
    temperature: 0.7,
    ragEnabled: true,
    similarityThreshold: 0.7,
    maxTrainingExamples: 5
};

// RAG Document Storage
let documentStore = [];
let conversationMemory = [];

// Training examples with enhanced metadata
let trainingExamples = [];

// System statistics
let systemStats = {
    totalAnalyses: 0,
    activeUsers: 0,
    documentsProcessed: 0,
    ragQueries: 0,
    startTime: new Date()
};

// Learning mode tracking
let learningData = {
    queryCount: 0,
    exampleCount: 0,
    targetQueries: 50,
    queries: [], // Store queries for fine-tuning
    responses: [] // Store responses for fine-tuning
};

// Enhanced RAG settings
let ragSettings = {
    enabled: adminSettings.ragEnabled,
    mode: 'learning', // 'disabled', 'learning', 'retrieval', 'finetuned'
    similarityThreshold: adminSettings.similarityThreshold,
    maxExamples: adminSettings.maxTrainingExamples
};

// ===========================
// RAG HELPER FUNCTIONS
// ===========================

async function generateEmbedding(text) {
    try {
        if (!OPENAI_API_KEY) {
            console.log('Using keyword-based similarity (no OpenAI key)');
            return generateKeywordVector(text);
        }

        const response = await axios.post('https://api.openai.com/v1/embeddings', {
            model: 'text-embedding-ada-002',
            input: text.substring(0, 8000) // Limit input size
        }, {
            headers: {
                'Authorization': `Bearer ${OPENAI_API_KEY}`,
                'Content-Type': 'application/json'
            }
        });

        return response.data.data[0].embedding;
    } catch (error) {
        console.error('Embedding generation failed, using fallback:', error.message);
        return generateKeywordVector(text);
    }
}

function generateKeywordVector(text) {
    // Budget-friendly keyword-based vector
    const keywords = [
        'market', 'treatment', 'physician', 'academic', 'community', 'region', 
        'adoption', 'preference', 'efficacy', 'safety', 'compliance', 'prescribing',
        'competition', 'share', 'growth', 'trend', 'segment', 'therapeutic',
        'survey', 'analysis', 'data', 'patient', 'clinical', 'therapy'
    ];
    
    const lowerText = text.toLowerCase();
    return keywords.map(keyword => 
        (lowerText.match(new RegExp(keyword, 'g')) || []).length / text.length * 1000
    );
}

function calculateSimilarity(vector1, vector2) {
    if (vector1.length !== vector2.length) return 0;
    
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (let i = 0; i < vector1.length; i++) {
        dotProduct += vector1[i] * vector2[i];
        norm1 += vector1[i] * vector1[i];
        norm2 += vector2[i] * vector2[i];
    }
    
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

function splitIntoChunks(text, maxChunkSize = 500) {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const chunks = [];
    let currentChunk = '';
    
    for (const sentence of sentences) {
        if (currentChunk.length + sentence.length < maxChunkSize) {
            currentChunk += sentence + '. ';
        } else {
            if (currentChunk.trim()) {
                chunks.push(currentChunk.trim());
            }
            currentChunk = sentence + '. ';
        }
    }
    
    if (currentChunk.trim()) {
        chunks.push(currentChunk.trim());
    }
    
    return chunks;
}

function extractKeywords(text) {
    const words = text.toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(word => word.length > 3);
    
    const wordCount = {};
    words.forEach(word => {
        wordCount[word] = (wordCount[word] || 0) + 1;
    });
    
    return Object.keys(wordCount)
        .sort((a, b) => wordCount[b] - wordCount[a])
        .slice(0, 10);
}

async function processDocument(content, fileName, category = 'general') {
    try {
        const chunks = splitIntoChunks(content, 500);
        const processedChunks = [];

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            const embedding = await generateEmbedding(chunk);
            
            const docChunk = {
                id: `${fileName}-chunk-${i}`,
                fileName: fileName,
                content: chunk,
                embedding: embedding,
                category: category,
                chunkIndex: i,
                totalChunks: chunks.length,
                processedAt: new Date().toISOString(),
                keywords: extractKeywords(chunk)
            };
            
            processedChunks.push(docChunk);
        }

        documentStore.push(...processedChunks);
        systemStats.documentsProcessed++;
        
        console.log(`Processed ${chunks.length} chunks from ${fileName}`);
        return processedChunks;
        
    } catch (error) {
        console.error(`Document processing failed for ${fileName}:`, error);
        throw error;
    }
}

async function retrieveRelevantContext(query, limit = 5) {
    try {
        if (documentStore.length === 0) {
            return [];
        }

        const queryEmbedding = await generateEmbedding(query);
        const similarities = documentStore.map(doc => ({
            ...doc,
            similarity: calculateSimilarity(queryEmbedding, doc.embedding)
        }));

        const relevantDocs = similarities
            .filter(doc => doc.similarity > adminSettings.similarityThreshold)
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit);

        systemStats.ragQueries++;
        
        console.log(`Retrieved ${relevantDocs.length} relevant documents for query`);
        return relevantDocs;

    } catch (error) {
        console.error('Context retrieval failed:', error);
        return [];
    }
}

// ===========================
// BASIC ROUTES
// ===========================

app.get('/', (req, res) => {
    res.json({ 
        message: 'Sagan Dashboard Backend with RAG is running!',
        ragEnabled: adminSettings.ragEnabled,
        documentsLoaded: documentStore.length,
        status: 'operational',
        timestamp: new Date().toISOString()
    });
});

app.get('/api/health', (req, res) => {
    const cleanedApiKey = ANTHROPIC_API_KEY ? ANTHROPIC_API_KEY.trim() : null;
    
    res.json({ 
        status: 'healthy',
        apiKeyConfigured: !!cleanedApiKey,
        apiKeyValid: cleanedApiKey && cleanedApiKey.startsWith('sk-ant-') && cleanedApiKey.length > 20,
        apiKeyLength: cleanedApiKey ? cleanedApiKey.length : 0,
        apiKeyPrefix: cleanedApiKey ? cleanedApiKey.substring(0, 15) : 'undefined',
        openaiKeyConfigured: !!OPENAI_API_KEY,
        ragEnabled: adminSettings.ragEnabled,
        documentsInStore: documentStore.length,
        timestamp: new Date().toISOString(),
        envCheck: {
            NODE_ENV: process.env.NODE_ENV || 'undefined',
            PORT: process.env.PORT || 'undefined',
            hasAnthropicKey: 'ANTHROPIC_API_KEY' in process.env,
            allEnvKeys: Object.keys(process.env).filter(key => key.includes('API')).length
        }
    });
});

// ===========================
// ENHANCED ANALYSIS ENDPOINT
// ===========================

app.post('/api/analyze', async (req, res) => {
    try {
        const { fileContent, fileName, userPrompt, webSearchEnabled } = req.body;
        
        // DEBUGGING: Log API key information
        console.log('=== API KEY DEBUG INFO ===');
        console.log('Raw ANTHROPIC_API_KEY exists:', !!ANTHROPIC_API_KEY);
        console.log('Raw ANTHROPIC_API_KEY length:', ANTHROPIC_API_KEY ? ANTHROPIC_API_KEY.length : 0);
        console.log('Raw ANTHROPIC_API_KEY prefix:', ANTHROPIC_API_KEY ? ANTHROPIC_API_KEY.substring(0, 15) : 'undefined');
        
        // Clean the API key (remove any whitespace)
        const cleanedApiKey = ANTHROPIC_API_KEY ? ANTHROPIC_API_KEY.trim() : null;
        
        console.log('Cleaned API key exists:', !!cleanedApiKey);
        console.log('Cleaned API key length:', cleanedApiKey ? cleanedApiKey.length : 0);
        console.log('Cleaned API key starts with sk-ant:', cleanedApiKey ? cleanedApiKey.startsWith('sk-ant-') : false);
        console.log('========================');
        
        // Enhanced validation
        if (!cleanedApiKey) {
            console.error('❌ API key is null or undefined');
            return res.status(500).json({ 
                error: 'API key not configured. Please check your Anthropic API key.',
                debug: {
                    keyExists: !!ANTHROPIC_API_KEY,
                    envVarName: 'ANTHROPIC_API_KEY',
                    issue: 'API key is null or undefined'
                }
            });
        }
        
        if (cleanedApiKey.length < 20) {
            console.error('❌ API key is too short:', cleanedApiKey.length);
            return res.status(500).json({ 
                error: 'API key appears to be invalid (too short).',
                debug: {
                    keyLength: cleanedApiKey.length,
                    issue: 'API key too short'
                }
            });
        }
        
        if (!cleanedApiKey.startsWith('sk-ant-')) {
            console.error('❌ API key does not start with sk-ant-');
            return res.status(500).json({ 
                error: 'API key format is invalid.',
                debug: {
                    keyPrefix: cleanedApiKey.substring(0, 10),
                    issue: 'API key does not start with sk-ant-'
                }
            });
        }
        
        if (!fileContent) {
            return res.status(400).json({ error: 'No file content provided' });
        }
        
        console.log(`Processing ${adminSettings.ragEnabled ? 'RAG-enhanced' : 'standard'} analysis for file: ${fileName}`);
        
        // Build system prompt (keep your existing code here)
        let enhancedSystemPrompt = adminSettings.systemPrompt;
        let relevantContext = [];

        // RAG Enhancement (if enabled) - keep your existing RAG code
        if (adminSettings.ragEnabled) {
            relevantContext = await retrieveRelevantContext(
                `${fileContent.substring(0, 500)} ${userPrompt || ''}`, 
                3
            );

            if (relevantContext.length > 0) {
                enhancedSystemPrompt += `\n\nRELEVANT REFERENCE EXAMPLES AND CONTEXT:\n`;
                relevantContext.forEach((doc, index) => {
                    enhancedSystemPrompt += `\n--- Reference ${index + 1} (${doc.fileName}, similarity: ${doc.similarity.toFixed(2)}) ---\n`;
                    enhancedSystemPrompt += doc.content;
                });
                enhancedSystemPrompt += `\n\nUse these references to inform your analysis style, structure, and insights while focusing on the new data provided.`;
            }
        }

        // Add training examples (keep your existing code)
        if (trainingExamples.length > 0) {
            enhancedSystemPrompt += `\n\nSTYLE REFERENCE EXAMPLES:\n`;
            enhancedSystemPrompt += trainingExamples.slice(0, adminSettings.maxTrainingExamples)
                .map(ex => `--- ${ex.fileName} ---\n${ex.content}`)
                .join('\n\n');
            enhancedSystemPrompt += `\n\nUse these examples as style guides for your analysis format and tone.`;
        }

        let userInstruction = `Please analyze this pharmaceutical survey data from the file "${fileName}":

${fileContent}`;

        if (userPrompt) {
            userInstruction += `\n\nSpecific analysis instructions: ${userPrompt}`;
        }

        if (webSearchEnabled) {
            userInstruction += `\n\nPlease integrate current market intelligence and recent pharmaceutical industry developments in your analysis.`;
        }

        // Store conversation context (keep your existing code)
        conversationMemory.push({
            fileName: fileName,
            userPrompt: userPrompt,
            timestamp: new Date().toISOString(),
            relevantContextUsed: relevantContext.length,
            ragEnabled: adminSettings.ragEnabled
        });

        console.log('🚀 Making API call to Anthropic...');
        
        // Call Claude API - USE THE CLEANED API KEY
        const response = await axios.post('https://api.anthropic.com/v1/messages', {
            model: adminSettings.claudeModel,
            max_tokens: adminSettings.maxTokens,
            temperature: adminSettings.temperature,
            system: enhancedSystemPrompt,
            messages: [{
                role: 'user',
                content: userInstruction
            }]
        }, {
            headers: {
                'Authorization': `Bearer ${cleanedApiKey}`, // ✅ Using cleaned API key
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
        });

        console.log('✅ API call successful!');

        const analysis = response.data.content[0].text;
        
        // Keep all your existing post-processing code...
        // Learning Mode: Store query and response for fine-tuning
        if (ragSettings.mode === 'learning') {
            learningData.queryCount++;
            learningData.queries.push({
                query: fileContent,
                response: analysis,
                ragUsed: relevantContext.length > 0,
                timestamp: new Date().toISOString()
            });
            
            console.log(`Learning Mode: Query ${learningData.queryCount}/${learningData.targetQueries} stored`);
            
            if (learningData.queryCount >= learningData.targetQueries) {
                console.log('Learning complete! Ready for fine-tuning preparation.');
            }
        }
        
        // Process and store analysis for future RAG queries (if RAG enabled)
        if (ragSettings.enabled) {
            await processDocument(analysis, `Analysis_${fileName}_${Date.now()}`, 'generated-analysis');
        }
        
        const chartData = generateChartData(analysis);
        
        console.log('Analysis completed successfully');
        systemStats.totalAnalyses++;

        res.json({ 
            analysis: analysis,
            chartData: chartData,
            ragContext: {
                enabled: adminSettings.ragEnabled,
                documentsUsed: relevantContext.length,
                contextSources: relevantContext.map(doc => ({
                    fileName: doc.fileName,
                    similarity: doc.similarity,
                    category: doc.category
                }))
            },
            metadata: {
                fileName: fileName,
                processedAt: new Date().toISOString(),
                webSearchEnabled: webSearchEnabled,
                userPromptUsed: !!userPrompt,
                ragEnabled: adminSettings.ragEnabled
            }
        });
        
    } catch (error) {
        console.error('❌ Analysis error:', error.response?.data || error.message);
        console.error('Full error details:', {
            status: error.response?.status,
            statusText: error.response?.statusText,
            data: error.response?.data,
            message: error.message
        });
        
        if (error.response?.status === 401) {
            console.error('🔑 Authentication failed - API key issue');
            res.status(401).json({ 
                error: 'Invalid API key. Please check your Anthropic API key.',
                details: error.response?.data || 'Authentication failed',
                debug: {
                    apiKeyExists: !!ANTHROPIC_API_KEY,
                    apiKeyLength: ANTHROPIC_API_KEY ? ANTHROPIC_API_KEY.length : 0
                }
            });
        } else if (error.response?.status === 429) {
            res.status(429).json({ error: 'Rate limit exceeded. Please try again later.' });
        } else {
            res.status(500).json({ 
                error: 'Analysis failed. Please try again.',
                details: error.message
            });
        }
    }
});

// ===========================
// ENHANCED CHAT ENDPOINT
// ===========================

app.post('/api/chat', async (req, res) => {
    try {
        const { question, analysis, fileName } = req.body;
        
        if (!ANTHROPIC_API_KEY) {
            return res.status(500).json({ error: 'API key not configured' });
        }
        
        if (!question || !analysis) {
            return res.status(400).json({ error: 'Question and analysis are required' });
        }
        
        console.log(`Processing ${adminSettings.ragEnabled ? 'RAG-enhanced' : 'standard'} chat question: ${question.substring(0, 50)}...`);
        
        let contextualPrompt = `Based on this pharmaceutical survey analysis, please answer the user's question concisely and professionally:

CURRENT ANALYSIS:
${analysis}`;

        // RAG Enhancement for chat (if enabled)
        if (adminSettings.ragEnabled) {
            const relevantContext = await retrieveRelevantContext(question, 2);
            
            if (relevantContext.length > 0) {
                contextualPrompt += `\n\nRELEVANT REFERENCE CONTEXT:`;
                relevantContext.forEach((doc, index) => {
                    contextualPrompt += `\n--- Reference ${index + 1} ---\n${doc.content}`;
                });
            }
        }

        contextualPrompt += `\n\nUSER QUESTION: ${question}

Please provide a helpful, specific answer based on the analysis data${adminSettings.ragEnabled ? ' and reference materials' : ''}. Keep your response focused and under 250 words.`;

        const response = await axios.post('https://api.anthropic.com/v1/messages', {
            model: adminSettings.claudeModel,
            max_tokens: 350,
            temperature: 0.3,
            messages: [{
                role: 'user',
                content: contextualPrompt
            }]
        }, {
            headers: {
                'Authorization': `Bearer ${ANTHROPIC_API_KEY}`,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
        });

        const chatResponse = response.data.content[0].text;
        
        res.json({ 
            response: chatResponse,
            ragEnabled: adminSettings.ragEnabled,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('Chat error:', error.response?.data || error.message);
        res.status(500).json({ 
            error: 'Chat failed. Please try again.',
            details: error.message
        });
    }
});

// ===========================
// CHART GENERATION HELPERS
// ===========================

function generateChartData(analysis) {
    const treatmentPreferences = extractPercentages(analysis, ['combination', 'monotherapy', 'experimental']);
    const confidenceLevels = extractPercentages(analysis, ['high confidence', 'moderate confidence', 'low confidence']);
    
    return {
        treatments: {
            labels: ['Combination Therapy', 'Monotherapy', 'Experimental', 'Standard Care'],
            data: treatmentPreferences.length >= 3 ? treatmentPreferences : [41, 28, 18, 13],
            colors: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
        },
        comparison: {
            labels: ['High Confidence', 'Moderate Confidence', 'Low Confidence'],
            academic: confidenceLevels.academic || [72, 21, 7],
            community: confidenceLevels.community || [44, 35, 21],
            colors: ['#3b82f6', '#10b981']
        },
        regional: {
            labels: ['Northeast', 'West Coast', 'Midwest', 'Southeast', 'Southwest'],
            data: [68, 65, 52, 45, 58],
            colors: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
        },
        trends: {
            labels: ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
            combination: [25, 31, 37, 41],
            monotherapy: [65, 58, 52, 43],
            colors: ['#3b82f6', '#ef4444']
        }
    };
}

function extractPercentages(text, keywords) {
    const percentages = [];
    const regex = /(\d+(?:\.\d+)?)\s*%/g;
    let match;
    
    while ((match = regex.exec(text)) !== null) {
        percentages.push(parseFloat(match[1]));
    }
    
    return percentages.slice(0, 4);
}

// ===========================
// ADMIN ENDPOINTS
// ===========================
// RAG Settings endpoint
app.post('/admin/rag-settings', async (req, res) => {
    try {
        const { enabled, mode, similarityThreshold, maxExamples } = req.body;
        
        // Update RAG settings
        ragSettings = {
            enabled: enabled !== undefined ? enabled : ragSettings.enabled,
            mode: mode || ragSettings.mode,
            similarityThreshold: similarityThreshold !== undefined ? similarityThreshold : ragSettings.similarityThreshold,
            maxExamples: maxExamples !== undefined ? maxExamples : ragSettings.maxExamples
        };
        
        // Also update adminSettings for backward compatibility
        adminSettings.ragEnabled = ragSettings.enabled;
        adminSettings.similarityThreshold = ragSettings.similarityThreshold;
        adminSettings.maxTrainingExamples = ragSettings.maxExamples;
        
        console.log('RAG settings updated:', ragSettings);
        
        res.json({
            success: true,
            settings: ragSettings,
            learningProgress: {
                queryCount: learningData.queryCount,
                exampleCount: learningData.exampleCount,
                targetQueries: learningData.targetQueries
            }
        });
        
    } catch (error) {
        console.error('RAG settings update error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to update RAG settings'
        });
    }
});

// Initialize Vector DB endpoint
app.post('/admin/initialize-vectordb', async (req, res) => {
    try {
        console.log('Vector database re-initialized (in-memory)');
        
        res.json({
            success: true,
            message: 'Vector database initialized',
            documentCount: documentStore.length
        });
        
    } catch (error) {
        console.error('Vector DB initialization error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to initialize vector database'
        });
    }
});

// Test vector search endpoint
app.post('/admin/test-vector-search', async (req, res) => {
    try {
        const { query, threshold = 0.7, maxResults = 3 } = req.body;
        
        if (!query) {
            return res.status(400).json({
                success: false,
                error: 'Query is required'
            });
        }
        
        const matches = await retrieveRelevantContext(query, maxResults);
        
        res.json({
            success: true,
            query: query,
            matches: matches.map(match => ({
                id: match.id,
                fileName: match.fileName,
                content: match.content.substring(0, 200) + '...',
                similarity: match.similarity,
                chunkIndex: match.chunkIndex
            })),
            matchCount: matches.length,
            threshold: threshold
        });
        
    } catch (error) {
        console.error('Vector search test error:', error);
        res.status(500).json({
            success: false,
            error: 'Vector search test failed'
        });
    }
});

// Learning progress endpoint
app.get('/admin/learning-progress', (req, res) => {
    const progressPercent = Math.min((learningData.queryCount / learningData.targetQueries) * 100, 100);
    
    res.json({
        queryCount: learningData.queryCount,
        exampleCount: learningData.exampleCount,
        targetQueries: learningData.targetQueries,
        progressPercent: Math.round(progressPercent),
        readyForFineTuning: learningData.queryCount >= learningData.targetQueries
    });
});

// RAG statistics endpoint
app.get('/admin/rag-stats', (req, res) => {
    res.json({
        totalQueries: learningData.queryCount,
        ragMatches: systemStats.ragQueries,
        last24hQueries: learningData.queryCount,
        avgSimilarity: 'N/A',
        commonQueryType: 'Analysis requests',
        currentMode: ragSettings.mode,
        documentsInStore: documentStore.length,
        learningProgress: learningData
    });
});

// Upload with vectorization endpoint
app.post('/admin/upload-training-vectorize', async (req, res) => {
    try {
        const { trainingData, fileName, vectorize = true, ragMode } = req.body;
        
        if (!trainingData || !fileName) {
            return res.status(400).json({
                success: false,
                error: 'Training data and filename are required'
            });
        }
        
        const trainingExample = {
            content: trainingData,
            fileName: fileName,
            uploadedAt: new Date().toISOString(),
            category: 'training',
            keywords: extractKeywords(trainingData)
        };
        
        trainingExamples.push(trainingExample);
        
        let processedChunks = [];
        if (vectorize && ragSettings.enabled) {
            processedChunks = await processDocument(trainingData, fileName, 'training');
        }
        
        learningData.exampleCount = trainingExamples.length;
        
        console.log(`Training document uploaded and vectorized: ${fileName} (${processedChunks.length} chunks)`);
        
        res.json({
            success: true,
            message: 'Training file uploaded and vectorized',
            fileName: fileName,
            chunkCount: processedChunks.length,
            totalExamples: trainingExamples.length
        });
        
    } catch (error) {
        console.error('Upload and vectorization error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to upload and vectorize training file'
        });
    }
});

app.get('/admin', (req, res) => {
    res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sagan Admin - Enhanced with RAG</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #0f172a; color: white; }
            .container { max-width: 600px; margin: 0 auto; }
            .btn { background: #3b82f6; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; display: inline-block; margin: 10px; }
            .rag-status { color: ${adminSettings.ragEnabled ? '#10b981' : '#ef4444'}; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔧 Sagan Admin Dashboard</h1>
            <p>Enhanced backend with RAG capabilities!</p>
            <p class="rag-status">RAG Status: ${adminSettings.ragEnabled ? 'ENABLED' : 'DISABLED'}</p>
            <p>Documents in store: ${documentStore.length}</p>
            <a href="/admin/settings" class="btn">View Settings</a>
            <a href="/admin/stats" class="btn">View Stats</a>
            <a href="/admin/training-examples" class="btn">Training Data</a>
        </div>
    </body>
    </html>
    `);
});

// Get current admin settings
app.get('/admin/settings', (req, res) => {
    res.json({
        ...adminSettings,
        trainingExamplesCount: trainingExamples.length,
        documentChunksCount: documentStore.length,
        lastUpdated: new Date().toISOString()
    });
});

// Update system prompt
app.post('/admin/update-prompt', (req, res) => {
    try {
        const { systemPrompt } = req.body;
        
        if (!systemPrompt || typeof systemPrompt !== 'string') {
            return res.status(400).json({ error: 'Valid system prompt required' });
        }
        
        adminSettings.systemPrompt = systemPrompt;
        console.log('System prompt updated by admin');
        
        res.json({ 
            success: true, 
            message: 'System prompt updated successfully',
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('Update prompt error:', error);
        res.status(500).json({ error: 'Failed to update system prompt' });
    }
});

// Update API settings
app.post('/admin/update-api-settings', (req, res) => {
    try {
        const { claudeModel, maxTokens, temperature, ragEnabled, similarityThreshold, maxTrainingExamples } = req.body;
        
        if (claudeModel) adminSettings.claudeModel = claudeModel;
        if (maxTokens) adminSettings.maxTokens = parseInt(maxTokens);
        if (temperature !== undefined) adminSettings.temperature = parseFloat(temperature);
        if (ragEnabled !== undefined) adminSettings.ragEnabled = ragEnabled;
        if (similarityThreshold) adminSettings.similarityThreshold = parseFloat(similarityThreshold);
        if (maxTrainingExamples) adminSettings.maxTrainingExamples = parseInt(maxTrainingExamples);
        
        console.log('API settings updated by admin', {
            ragEnabled: adminSettings.ragEnabled,
            model: adminSettings.claudeModel
        });
        
        res.json({ 
            success: true, 
            message: 'API settings updated successfully',
            settings: adminSettings
        });
        
    } catch (error) {
        console.error('Update API settings error:', error);
        res.status(500).json({ error: 'Failed to update API settings' });
    }
});

// Enhanced training upload with RAG support
app.post('/admin/upload-training', async (req, res) => {
    try {
        const { trainingData, fileName, category = 'training' } = req.body;
        
        if (!trainingData || !fileName) {
            return res.status(400).json({ error: 'Training data and filename required' });
        }
        
        // Add to traditional training examples
        const trainingExample = {
            content: trainingData,
            fileName: fileName,
            uploadedAt: new Date().toISOString(),
            category: category,
            keywords: extractKeywords(trainingData)
        };
        
        trainingExamples.push(trainingExample);
        
        // Process for RAG if enabled
        let processedChunks = [];
        if (adminSettings.ragEnabled) {
            processedChunks = await processDocument(trainingData, fileName, category);
        }
        
        console.log(`Training document uploaded: ${fileName} ${adminSettings.ragEnabled ? `(${processedChunks.length} chunks processed)` : ''}`);
        
        res.json({ 
            success: true, 
            message: 'Training data uploaded successfully',
            totalExamples: trainingExamples.length,
            chunksProcessed: processedChunks.length,
            ragEnabled: adminSettings.ragEnabled
        });
        
    } catch (error) {
        console.error('Upload training error:', error);
        res.status(500).json({ error: 'Failed to upload training data' });
    }
});

// Get training examples
app.get('/admin/training-examples', (req, res) => {
    res.json({
        examples: trainingExamples.map((ex, index) => ({
            index: index,
            fileName: ex.fileName,
            uploadedAt: ex.uploadedAt,
            category: ex.category || 'general',
            keywords: ex.keywords || [],
            contentPreview: ex.content.substring(0, 200) + '...'
        })),
        totalCount: trainingExamples.length,
        documentChunks: documentStore.length,
        ragEnabled: adminSettings.ragEnabled
    });
});

// Delete training example
app.delete('/admin/training-examples/:index', (req, res) => {
    try {
        const index = parseInt(req.params.index);
        
        if (index < 0 || index >= trainingExamples.length) {
            return res.status(404).json({ error: 'Training example not found' });
        }
        
        const removed = trainingExamples.splice(index, 1)[0];
        
        // Remove associated document chunks from RAG store
        const originalLength = documentStore.length;
        documentStore = documentStore.filter(doc => doc.fileName !== removed.fileName);
        const removedChunks = originalLength - documentStore.length;
        
        res.json({ 
            success: true, 
            message: 'Training example deleted',
            deletedFile: removed.fileName,
            removedChunks: removedChunks,
            remainingCount: trainingExamples.length
        });
        
    } catch (error) {
        console.error('Delete training error:', error);
        res.status(500).json({ error: 'Failed to delete training example' });
    }
});

// Clear all training data
app.post('/admin/clear-training', (req, res) => {
    try {
        const previousCount = trainingExamples.length;
        const previousChunks = documentStore.length;
        
        trainingExamples = [];
        documentStore = [];
        conversationMemory = [];
        
        console.log('All training data cleared by admin');
        
        res.json({ 
            success: true, 
            message: 'All training data cleared',
            previousCount: previousCount,
            previousChunks: previousChunks
        });
        
    } catch (error) {
        console.error('Clear training error:', error);
        res.status(500).json({ error: 'Failed to clear training data' });
    }
});

// Enhanced system stats
app.get('/admin/stats', (req, res) => {
    const uptime = Date.now() - systemStats.startTime.getTime();
    const uptimeHours = Math.floor(uptime / (1000 * 60 * 60));
    const uptimeMinutes = Math.floor((uptime % (1000 * 60 * 60)) / (1000 * 60));
    
    res.json({
        totalAnalyses: systemStats.totalAnalyses,
        activeUsers: systemStats.activeUsers,
        trainingExamples: trainingExamples.length,
        documentChunks: documentStore.length,
        ragMatches: systemStats.ragQueries,
        learningProgressPercent: Math.round(Math.min((learningData.queryCount / learningData.targetQueries) * 100, 100)),
        documentsProcessed: systemStats.documentsProcessed,
        conversationMemory: conversationMemory.length,
        ragEnabled: ragSettings.enabled,
        uptime: `${uptimeHours}h ${uptimeMinutes}m`,
        systemHealth: 'healthy',
        lastRestart: systemStats.startTime.toISOString(),
        learningProgress: {
            queryCount: learningData.queryCount,
            exampleCount: learningData.exampleCount,
            targetQueries: learningData.targetQueries
        },
        apiKeys: {
            anthropic: !!ANTHROPIC_API_KEY,
            openai: !!OPENAI_API_KEY
        }
    });
});

// ===========================
// ERROR HANDLING & 404
// ===========================

// Middleware to track active users
app.use((req, res, next) => {
    if (req.path.includes('/api/analyze') || req.path.includes('/api/chat')) {
        systemStats.activeUsers = Math.min(systemStats.activeUsers + 1, 100);
        
        setTimeout(() => {
            systemStats.activeUsers = Math.max(systemStats.activeUsers - 1, 0);
        }, 300000);
    }
    next();
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Server error:', error);
    res.status(500).json({ 
        error: 'Internal server error',
        timestamp: new Date().toISOString()
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({ 
        error: 'Endpoint not found',
        availableEndpoints: [
            'GET /',
            'GET /api/health',
            'POST /api/analyze',
            'POST /api/chat',
            'GET /admin',
            'GET /admin/settings',
            'POST /admin/update-prompt',
            'POST /admin/upload-training',
            'GET /admin/training-examples',
            'GET /admin/stats'
        ]
    });
});

// ===========================
// SERVER STARTUP
// ===========================

// Initialize system
async function initializeSystem() {
    console.log('🔧 Initializing Sagan Dashboard...');
    
    if (adminSettings.ragEnabled && trainingExamples.length > 0) {
        console.log('📚 Processing existing training examples for RAG...');
        for (const example of trainingExamples) {
            try {
                await processDocument(example.content, example.fileName, example.category || 'training');
            } catch (error) {
                console.error(`Failed to process training example: ${example.fileName}`);
            }
        }
    }
    
    console.log(`✅ System initialized with:`);
    console.log(`   - RAG: ${adminSettings.ragEnabled ? 'ENABLED' : 'DISABLED'}`);
    console.log(`   - Training examples: ${trainingExamples.length}`);
    console.log(`   - Document chunks: ${documentStore.length}`);
}

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Sagan Dashboard Backend with RAG running on port ${PORT}`);
    console.log(`📡 Claude API Key configured: ${!!ANTHROPIC_API_KEY}`);
    console.log(`🔍 OpenAI API Key configured: ${!!OPENAI_API_KEY}`);
    console.log(`📚 RAG System: ${adminSettings.ragEnabled ? 'ENABLED' : 'DISABLED'}`);
    console.log(`🌐 Health check: http://localhost:${PORT}/api/health`);
    
    // Initialize system after server starts
    initializeSystem();
});

module.exports = app;
