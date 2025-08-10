// server.js - Sagan Dashboard Backend Server
// ==========================================

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const app = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Environment variables
const PORT = process.env.PORT || 3000;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY; // Your API key goes here

// Basic route
app.get('/', (req, res) => {
    res.json({ 
        message: 'Sagan Dashboard Backend is running!',
        status: 'operational',
        timestamp: new Date().toISOString()
    });
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'healthy',
        apiKeyConfigured: !!ANTHROPIC_API_KEY,
        timestamp: new Date().toISOString()
    });
});

// Main analysis endpoint
app.post('/api/analyze', async (req, res) => {
    try {
        const { fileContent, fileName, userPrompt, webSearchEnabled } = req.body;
        
        if (!ANTHROPIC_API_KEY) {
            return res.status(500).json({ 
                error: 'API key not configured. Please set ANTHROPIC_API_KEY environment variable.' 
            });
        }
        
        if (!fileContent) {
            return res.status(400).json({ error: 'No file content provided' });
        }
        
        console.log(`Processing analysis for file: ${fileName}`);
        
        // Construct the prompt for Claude
        const systemPrompt = `You are an expert pharmaceutical market research analyst specializing in physician survey data analysis and executive summary generation.

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

Format your response as a professional executive summary with clear headers and bullet points.`;

        let userInstruction = `Please analyze this pharmaceutical survey data from the file "${fileName}":

${fileContent}`;

        if (userPrompt) {
            userInstruction += `\n\nSpecific analysis instructions: ${userPrompt}`;
        }

        if (webSearchEnabled) {
            userInstruction += `\n\nPlease integrate current market intelligence and recent pharmaceutical industry developments in your analysis.`;
        }

        // Call Claude API
        const response = await axios.post('https://api.anthropic.com/v1/messages', {
            model: 'claude-3-5-sonnet-20241022',
            max_tokens: 4000,
            temperature: 0.7,
            system: systemPrompt,
            messages: [{
                role: 'user',
                content: userInstruction
            }]
        }, {
            headers: {
                'x-api-key': ANTHROPIC_API_KEY,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
        });

        const analysis = response.data.content[0].text;
        
        // Generate sample chart data based on analysis
        const chartData = generateChartData(analysis);
        
        console.log('Analysis completed successfully');
        
        res.json({ 
            analysis: analysis,
            chartData: chartData,
            metadata: {
                fileName: fileName,
                processedAt: new Date().toISOString(),
                webSearchEnabled: webSearchEnabled,
                userPromptUsed: !!userPrompt
            }
        });
        
    } catch (error) {
        console.error('Analysis error:', error.response?.data || error.message);
        
        if (error.response?.status === 401) {
            res.status(401).json({ error: 'Invalid API key. Please check your Anthropic API key.' });
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

// Chat endpoint for follow-up questions
app.post('/api/chat', async (req, res) => {
    try {
        const { question, analysis, fileName } = req.body;
        
        if (!ANTHROPIC_API_KEY) {
            return res.status(500).json({ error: 'API key not configured' });
        }
        
        if (!question || !analysis) {
            return res.status(400).json({ error: 'Question and analysis are required' });
        }
        
        console.log(`Processing chat question: ${question.substring(0, 50)}...`);
        
        const chatPrompt = `Based on this pharmaceutical survey analysis, please answer the user's question concisely and professionally:

ANALYSIS:
${analysis}

USER QUESTION: ${question}

Please provide a helpful, specific answer based on the analysis data. Keep your response focused and under 200 words.`;

        const response = await axios.post('https://api.anthropic.com/v1/messages', {
            model: 'claude-3-5-sonnet-20241022',
            max_tokens: 300,
            temperature: 0.3,
            messages: [{
                role: 'user',
                content: chatPrompt
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

// Helper function to generate chart data
function generateChartData(analysis) {
    // Extract percentages and create chart data
    // This is a simplified version - you can enhance this to parse actual data from the analysis
    
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
            data: [68, 65, 52, 45, 58], // Could be extracted from analysis
            colors: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
        },
        trends: {
            labels: ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
            combination: [25, 31, 37, 41], // Could be extracted from analysis
            monotherapy: [65, 58, 52, 43],
            colors: ['#3b82f6', '#ef4444']
        }
    };
}

// Helper function to extract percentages from analysis text
function extractPercentages(text, keywords) {
    const percentages = [];
    const regex = /(\d+(?:\.\d+)?)\s*%/g;
    let match;
    
    while ((match = regex.exec(text)) !== null) {
        percentages.push(parseFloat(match[1]));
    }
    
    return percentages.slice(0, 4); // Return first 4 percentages found
}

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
            'POST /api/chat'
        ]
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Sagan Dashboard Backend running on port ${PORT}`);
    console.log(`📡 API Key configured: ${!!ANTHROPIC_API_KEY}`);
    console.log(`🌐 Health check: http://localhost:${PORT}/api/health`);
});

module.exports = app;
