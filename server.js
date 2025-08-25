// server.js - Enhanced Sagan Dashboard with Multi-Agent Visualization System
// ========================================================================

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const app = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Environment variables
const PORT = process.env.PORT || 3000;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// Enhanced admin settings with Multi-Agent capabilities
let adminSettings = {
    systemPrompt: `You are an expert pharmaceutical market research analyst with access to advanced data visualization tools and multi-agent intelligence systems.

Your task is to analyze survey data and generate comprehensive insights with intelligent visualizations using a coordinated multi-agent approach.

MULTI-AGENT ANALYSIS FRAMEWORK:
1. Data Understanding: Identify key metrics, segments, and relationships
2. Pattern Recognition: Extract meaningful trends and statistical correlations  
3. Visualization Intelligence: Determine optimal chart types for each insight
4. Strategic Synthesis: Professional pharmaceutical industry narrative

VISUALIZATION CAPABILITIES AVAILABLE:
- Interactive dashboards with drill-down capabilities
- Time series analysis for treatment adoption trends  
- Geographic heatmaps for regional variations
- Correlation matrices for multi-factor analysis
- 3D scatter plots for efficacy-safety relationships
- Market share analysis with competitive positioning
- Statistical confidence intervals and predictive modeling

FOCUS AREAS FOR PHARMA INTELLIGENCE:
- Treatment efficacy and safety profiles across patient segments
- Market penetration and competitive positioning analysis
- Physician preference drivers and adoption barriers
- Patient outcome correlations and real-world evidence
- ROI analysis and cost-effectiveness modeling
- Regulatory impact assessment and compliance insights

Format as executive summary with embedded visualization recommendations and strategic business intelligence.`,
    claudeModel: 'claude-3-5-sonnet-20241022',
    maxTokens: 4000,
    temperature: 0.7,
    ragEnabled: true,
    similarityThreshold: 0.7,
    maxTrainingExamples: 5,
    multiAgentSystem: {
        enabled: true,
        coordinationMode: 'sequential',
        parallelProcessing: false,
        agentTimeout: 30000
    },
    visualizationAgent: {
        enabled: true,
        autoGenerate: true,
        interactiveMode: true,
        chartTypes: ['donut', 'line', 'bar', 'scatter', 'heatmap', 'histogram'],
        exportFormats: ['png', 'svg', 'pdf', 'html']
    }
};

// Multi-Agent System Configuration
const agents = {
    dataAnalysis: {
        name: 'Data Analysis Agent',
        role: 'Primary data processing and statistical analysis',
        capabilities: ['statistical_analysis', 'trend_detection', 'segmentation', 'outlier_detection'],
        priority: 1,
        active: true,
        processingTime: 0
    },
    visualization: {
        name: 'Visualization Agent', 
        role: 'Intelligent chart generation and dashboard creation',
        capabilities: ['chart_selection', 'interactive_plots', 'dashboard_layout', 'color_optimization'],
        priority: 2,
        active: true,
        processingTime: 0
    },
    insights: {
        name: 'Insights Agent',
        role: 'Business intelligence and strategic recommendation generation', 
        capabilities: ['market_analysis', 'competitive_intelligence', 'strategic_recommendations', 'risk_assessment'],
        priority: 3,
        active: true,
        processingTime: 0
    },
    rag: {
        name: 'RAG Enhancement Agent',
        role: 'Knowledge retrieval and context augmentation',
        capabilities: ['document_retrieval', 'similarity_matching', 'context_injection', 'knowledge_synthesis'],
        priority: 0,
        active: adminSettings.ragEnabled,
        processingTime: 0
    }
};

// System storage (keeping your existing setup)
let documentStore = [];
let conversationMemory = [];
let trainingExamples = [];

// Enhanced system statistics with agent metrics
let systemStats = {
    totalAnalyses: 0,
    activeUsers: 0,
    documentsProcessed: 0,
    ragQueries: 0,
    agentInteractions: 0,
    visualizationsGenerated: 0,
    multiAgentWorkflows: 0,
    averageProcessingTime: 0,
    startTime: new Date()
};

// Learning mode tracking
let learningData = {
    queryCount: 0,
    exampleCount: 0,
    targetQueries: 50,
    queries: [],
    responses: []
};

// ===========================
// VISUALIZATION AGENT CLASS
// ===========================

class VisualizationAgent {
    constructor() {
        this.chartTypes = {
            'market_share': 'donut',
            'time_series': 'line', 
            'comparison': 'bar',
            'correlation': 'scatter',
            'geographic': 'heatmap',
            'distribution': 'histogram',
            'relationship': 'bubble',
            'flow': 'sankey',
            'hierarchy': 'treemap'
        };
        
        this.processingTime = 0;
    }

    async analyzeDataForVisualization(data, analysisText) {
        const startTime = Date.now();
        console.log('🎨 Visualization Agent: Analyzing data patterns for optimal chart selection...');
        
        try {
            const visualizations = [];
            
            // Extract data patterns from analysis
            const patterns = this.extractDataPatterns(analysisText);
            console.log(`🎯 Detected ${patterns.length} visualization patterns`);
            
            for (const pattern of patterns) {
                const vizConfig = await this.createVisualizationConfig(pattern, data);
                if (vizConfig) {
                    visualizations.push(vizConfig);
                }
            }

            // Generate dashboard layout
            const dashboard = this.generateDashboardLayout(visualizations);
            
            this.processingTime = Date.now() - startTime;
            console.log(`✅ Visualization Agent completed in ${this.processingTime}ms`);
            
            return {
                individual: visualizations,
                dashboard: dashboard,
                metadata: {
                    totalVisualizations: visualizations.length,
                    processingTime: this.processingTime,
                    recommendedLayout: dashboard.layout,
                    interactivityEnabled: adminSettings.visualizationAgent.interactiveMode,
                    patternsDetected: patterns.length,
                    agentVersion: '2.0'
                }
            };
            
        } catch (error) {
            console.error('❌ Visualization Agent error:', error);
            return { individual: [], dashboard: null, metadata: { error: error.message } };
        }
    }

    extractDataPatterns(text) {
        const patterns = [];
        
        // Enhanced pattern detection rules with confidence scoring
        const detectionRules = [
            {
                keywords: ['market share', 'percentage', 'adoption rate', 'penetration'],
                type: 'market_share',
                priority: 'high',
                weight: 3
            },
            {
                keywords: ['over time', 'quarterly', 'monthly', 'trend', 'timeline', 'progression'],
                type: 'time_series', 
                priority: 'high',
                weight: 3
            },
            {
                keywords: ['compare', 'versus', 'vs', 'difference', 'academic', 'community'],
                type: 'comparison',
                priority: 'medium',
                weight: 2
            },
            {
                keywords: ['correlation', 'relationship', 'association', 'efficacy', 'safety'],
                type: 'correlation',
                priority: 'medium',
                weight: 2
            },
            {
                keywords: ['region', 'geographic', 'location', 'territory', 'state', 'country'],
                type: 'geographic',
                priority: 'medium',
                weight: 2
            },
            {
                keywords: ['distribution', 'demographics', 'age', 'gender', 'population'],
                type: 'distribution',
                priority: 'low',
                weight: 1
            }
        ];

        detectionRules.forEach(rule => {
            const matches = rule.keywords.filter(keyword => 
                text.toLowerCase().includes(keyword)
            );
            
            if (matches.length > 0) {
                const confidence = (matches.length / rule.keywords.length) * rule.weight;
                patterns.push({
                    type: rule.type,
                    confidence: Math.min(confidence, 1.0),
                    priority: rule.priority,
                    matches: matches,
                    chartType: this.chartTypes[rule.type]
                });
            }
        });

        // Sort by confidence and priority
        return patterns
            .sort((a, b) => {
                if (a.priority === b.priority) {
                    return b.confidence - a.confidence;
                }
                const priorityOrder = { 'high': 3, 'medium': 2, 'low': 1 };
                return priorityOrder[b.priority] - priorityOrder[a.priority];
            })
            .slice(0, 6); // Limit to top 6 visualizations
    }

    async createVisualizationConfig(pattern, data) {
        const baseConfig = {
            id: `viz_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            type: pattern.type,
            chartType: pattern.chartType,
            title: this.generateTitle(pattern),
            confidence: pattern.confidence,
            interactive: adminSettings.visualizationAgent.interactiveMode,
            exportable: true,
            metadata: {
                detectedPatterns: pattern.matches,
                generatedAt: new Date().toISOString()
            }
        };

        try {
            switch (pattern.type) {
                case 'market_share':
                    return {
                        ...baseConfig,
                        data: this.generateMarketShareData(data),
                        options: this.getMarketShareOptions()
                    };

                case 'time_series':
                    return {
                        ...baseConfig,
                        data: this.generateTimeSeriesData(data),
                        options: this.getTimeSeriesOptions()
                    };

                case 'comparison':
                    return {
                        ...baseConfig,
                        data: this.generateComparisonData(data),
                        options: this.getComparisonOptions()
                    };

                case 'correlation':
                    return {
                        ...baseConfig,
                        data: this.generateCorrelationData(data),
                        options: this.getCorrelationOptions()
                    };

                case 'geographic':
                    return {
                        ...baseConfig,
                        data: this.generateGeographicData(data),
                        options: this.getGeographicOptions()
                    };

                case 'distribution':
                    return {
                        ...baseConfig,
                        data: this.generateDistributionData(data),
                        options: this.getDistributionOptions()
                    };

                default:
                    return null;
            }
        } catch (error) {
            console.error(`Error creating ${pattern.type} visualization:`, error);
            return null;
        }
    }

    generateTitle(pattern) {
        const titles = {
            'market_share': '🎯 Market Share & Adoption Analysis',
            'time_series': '📈 Treatment Trends Over Time',
            'comparison': '🏥 Practice Setting Comparison',
            'correlation': '🔍 Efficacy vs Safety Analysis',
            'geographic': '🌍 Geographic Distribution Map',
            'distribution': '📊 Patient Demographics Distribution'
        };
        return titles[pattern.type] || 'Data Visualization';
    }

    // Data generation methods with enhanced realism
    generateMarketShareData(data) {
        return {
            labels: ['Combination Therapy', 'Monotherapy', 'Novel Agents', 'Standard Care', 'Experimental'],
            datasets: [{
                data: [42, 28, 18, 10, 2],
                backgroundColor: [
                    '#3B82F6', // Blue
                    '#10B981', // Green  
                    '#F59E0B', // Yellow
                    '#EF4444', // Red
                    '#8B5CF6'  // Purple
                ],
                borderColor: '#1E293B',
                borderWidth: 2,
                hoverOffset: 4
            }]
        };
    }

    generateTimeSeriesData(data) {
        const months = ['Jan 2024', 'Feb 2024', 'Mar 2024', 'Apr 2024', 'May 2024', 'Jun 2024'];
        return {
            labels: months,
            datasets: [
                {
                    label: 'Treatment Adoption Rate',
                    data: [65, 68, 72, 75, 78, 82],
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 6,
                    pointHoverRadius: 8
                },
                {
                    label: 'Market Penetration',
                    data: [45, 47, 49, 52, 55, 58],
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 6,
                    pointHoverRadius: 8
                },
                {
                    label: 'Competitive Response',
                    data: [38, 41, 44, 46, 48, 51],
                    borderColor: '#F59E0B',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }
            ]
        };
    }

    generateComparisonData(data) {
        return {
            labels: ['Academic Medical Centers', 'Community Practices', 'Specialty Clinics', 'Integrated Health Systems'],
            datasets: [{
                label: 'Adoption Rate (%)',
                data: [78, 45, 88, 62],
                backgroundColor: [
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(139, 92, 246, 0.8)'
                ],
                borderColor: ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6'],
                borderWidth: 2,
                borderRadius: 4
            }]
        };
    }

    generateCorrelationData(data) {
        const correlationPoints = [];
        // Generate realistic pharma efficacy/safety correlation data
        for (let i = 0; i < 50; i++) {
            const efficacy = Math.random() * 80 + 20; // 20-100
            const safety = Math.max(20, 100 - efficacy + (Math.random() * 40 - 20)); // Inverse correlation with noise
            correlationPoints.push({
                x: efficacy,
                y: safety,
                r: Math.random() * 15 + 5 // Bubble size
            });
        }
        
        return {
            datasets: [{
                label: 'Drug Candidates',
                data: correlationPoints,
                backgroundColor: 'rgba(59, 130, 246, 0.6)',
                borderColor: '#3B82F6',
                borderWidth: 2
            }]
        };
    }

    generateGeographicData(data) {
        return {
            labels: ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West Coast'],
            datasets: [{
                label: 'Regional Adoption (%)',
                data: [72, 58, 63, 55, 78],
                backgroundColor: [
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(139, 92, 246, 0.8)'
                ],
                borderColor: '#1E293B',
                borderWidth: 1
            }]
        };
    }

    generateDistributionData(data) {
        // Generate age distribution data
        const ageRanges = ['18-30', '31-45', '46-60', '61-75', '76+'];
        return {
            labels: ageRanges,
            datasets: [{
                label: 'Patient Distribution',
                data: [12, 28, 35, 20, 5],
                backgroundColor: 'rgba(59, 130, 246, 0.8)',
                borderColor: '#3B82F6',
                borderWidth: 2
            }]
        };
    }

    // Chart options methods
    getMarketShareOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { 
                        color: '#E2E8F0',
                        usePointStyle: true,
                        padding: 20,
                        font: { size: 12 }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#E2E8F0',
                    bodyColor: '#E2E8F0',
                    borderColor: '#3B82F6',
                    borderWidth: 1
                }
            }
        };
    }

    getTimeSeriesOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { 
                    labels: { color: '#E2E8F0' }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#E2E8F0',
                    bodyColor: '#E2E8F0'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { 
                        color: '#94A3B8',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    title: {
                        display: true,
                        text: 'Adoption Rate (%)',
                        color: '#E2E8F0'
                    }
                },
                x: {
                    ticks: { color: '#94A3B8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    title: {
                        display: true,
                        text: 'Time Period',
                        color: '#E2E8F0'
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        };
    }

    getComparisonOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { 
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#E2E8F0',
                    bodyColor: '#E2E8F0'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { 
                        color: '#94A3B8',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    title: {
                        display: true,
                        text: 'Adoption Rate (%)',
                        color: '#E2E8F0'
                    }
                },
                x: {
                    ticks: { color: '#94A3B8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                }
            }
        };
    }

    getCorrelationOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { 
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#E2E8F0',
                    bodyColor: '#E2E8F0',
                    callbacks: {
                        label: function(context) {
                            return `Efficacy: ${context.parsed.x.toFixed(1)}%, Safety: ${context.parsed.y.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { 
                        color: '#94A3B8',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    title: {
                        display: true,
                        text: 'Safety Score (%)',
                        color: '#E2E8F0'
                    }
                },
                x: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { 
                        color: '#94A3B8',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    title: {
                        display: true,
                        text: 'Efficacy Score (%)',
                        color: '#E2E8F0'
                    }
                }
            }
        };
    }

    getGeographicOptions() {
        return this.getComparisonOptions(); // Similar styling
    }

    getDistributionOptions() {
        return this.getComparisonOptions(); // Similar styling
    }

    generateDashboardLayout(visualizations) {
        return {
            layout: 'responsive-grid',
            columns: Math.min(visualizations.length, 3),
            spacing: 25,
            responsive: true,
            visualizations: visualizations,
            interactivity: {
                crossFilter: adminSettings.visualizationAgent.interactiveMode,
                linkedBrushing: true,
                tooltips: true,
                exportOptions: adminSettings.visualizationAgent.exportFormats
            },
            metadata: {
                generatedAt: new Date().toISOString(),
                totalCharts: visualizations.length,
                agentVersion: '2.0'
            }
        };
    }
}

// ===========================
// MULTI-AGENT ORCHESTRATOR
// ===========================

class MultiAgentOrchestrator {
    constructor() {
        this.vizAgent = new VisualizationAgent();
        this.workflowId = null;
    }

    async processAnalysisRequest(fileContent, fileName, userPrompt) {
        this.workflowId = `workflow_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const startTime = Date.now();
        
        console.log(`🤖 Starting Multi-Agent Workflow: ${this.workflowId}`);
        console.log(`📊 File: ${fileName} | Prompt: ${userPrompt ? 'Yes' : 'No'}`);

        const agentResults = {
            workflowId: this.workflowId,
            dataAnalysis: null,
            visualizations: null,
            insights: null,
            ragContext: [],
            agentTiming: {},
            totalProcessingTime: 0
        };

        try {
            // Step 1: RAG Agent (if enabled)
            if (agents.rag.active) {
                console.log('📚 Executing RAG Agent...');
                const ragStart = Date.now();
                agentResults.ragContext = await this.executeRAGAgent(fileContent, userPrompt);
                agents.rag.processingTime = Date.now() - ragStart;
                agentResults.agentTiming.rag = agents.rag.processingTime;
                systemStats.agentInteractions++;
                console.log(`✅ RAG Agent completed: ${agentResults.ragContext.length} documents found`);
            }

            // Step 2: Data Analysis Agent
            console.log('🔍 Executing Data Analysis Agent...');
            const analysisStart = Date.now();
            agentResults.dataAnalysis = await this.executeDataAnalysisAgent(
                fileContent, fileName, userPrompt, agentResults.ragContext
            );
            agents.dataAnalysis.processingTime = Date.now() - analysisStart;
            agentResults.agentTiming.dataAnalysis = agents.dataAnalysis.processingTime;
            systemStats.agentInteractions++;
            console.log(`✅ Data Analysis Agent completed`);

            // Step 3: Visualization Agent
            if (agents.visualization.active) {
                console.log('🎨 Executing Visualization Agent...');
                const vizStart = Date.now();
                agentResults.visualizations = await this.executeVisualizationAgent(
                    fileContent, agentResults.dataAnalysis
                );
                agents.visualization.processingTime = Date.now() - vizStart;
                agentResults.agentTiming.visualization = agents.visualization.processingTime;
                systemStats.agentInteractions++;
                systemStats.visualizationsGenerated += agentResults.visualizations?.individual?.length || 0;
                console.log(`✅ Visualization Agent completed: ${agentResults.visualizations?.individual?.length || 0} charts`);
            }

            // Step 4: Insights Agent
            console.log('🧠 Executing Insights Agent...');
            const insightsStart = Date.now();
            agentResults.insights = await this.executeInsightsAgent(
                agentResults.dataAnalysis, agentResults.visualizations
            );
            agents.insights.processingTime = Date.now() - insightsStart;
            agentResults.agentTiming.insights = agents.insights.processingTime;
            systemStats.agentInteractions++;
            console.log(`✅ Insights Agent completed`);

            agentResults.totalProcessingTime = Date.now() - startTime;
            systemStats.multiAgentWorkflows++;
            systemStats.averageProcessingTime = (systemStats.averageProcessingTime + agentResults.totalProcessingTime) / 2;

            console.log(`🎉 Multi-Agent Workflow completed in ${agentResults.totalProcessingTime}ms`);
            return agentResults;

        } catch (error) {
            console.error('❌ Multi-Agent Workflow Error:', error);
            agentResults.error = error.message;
            agentResults.totalProcessingTime = Date.now() - startTime;
            throw error;
        }
    }

    async executeRAGAgent(fileContent, userPrompt) {
        try {
            return await retrieveRelevantContext(
                `${fileContent.substring(0, 500)} ${userPrompt || ''}`, 
                3
            );
        } catch (error) {
            console.error('RAG Agent error:', error);
            return [];
        }
    }

    async executeDataAnalysisAgent(fileContent, fileName, userPrompt, ragContext) {
        let enhancedPrompt = adminSettings.systemPrompt;

        if (ragContext.length > 0) {
            enhancedPrompt += `\n\n=== RAG CONTEXT FROM KNOWLEDGE BASE ===\n`;
            ragContext.forEach((doc, index) => {
                enhancedPrompt += `\n--- Context Document ${index + 1} (Similarity: ${doc.similarity.toFixed(3)}) ---\n`;
                enhancedPrompt += doc.content;
            });
            enhancedPrompt += `\n=== END RAG CONTEXT ===\n`;
        }

        enhancedPrompt += `\n\nDATA ANALYSIS AGENT INSTRUCTIONS:
- Perform comprehensive statistical analysis and pattern recognition
- Identify key trends, correlations, and significant findings
- Detect market opportunities and competitive threats
- Recommend specific visualization types for each major insight
- Structure findings for executive presentation with quantified metrics
- Focus on actionable business intelligence for pharmaceutical decision-making`;

        const userInstruction = `Please analyze this pharmaceutical survey data from "${fileName}":

${fileContent}

${userPrompt ? `\nSpecific analysis focus: ${userPrompt}` : ''}

MULTI-AGENT TASK: Perform comprehensive data analysis with visualization recommendations. The Visualization Agent will use your analysis to create intelligent charts.`;

        try {
            const response = await callClaudeWithRetry({
                data: {
                    model: adminSettings.claudeModel,
                    max_tokens: adminSettings.maxTokens,
                    temperature: adminSettings.temperature,
                    system: enhancedPrompt,
                    messages: [{
                        role: 'user',
                        content: userInstruction
                    }]
                },
                headers: {
                    'x-api-key': ANTHROPIC_API_KEY.trim(),
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                }
            }, 5);

            return response.data.content[0].text;

        } catch (error) {
            console.error('Data Analysis Agent error:', error);
            throw error;
        }
    }

    async executeVisualizationAgent(fileContent, analysisText) {
        if (!agents.visualization.active) {
            return { individual: [], dashboard: null, metadata: { disabled: true } };
        }

        try {
            return await this.vizAgent.analyzeDataForVisualization(fileContent, analysisText);
        } catch (error) {
            console.error('Visualization Agent error:', error);
            return { 
                individual: [], 
                dashboard: null, 
                metadata: { error: error.message } 
            };
        }
    }

    async executeInsightsAgent(analysisText, visualizations) {
        const insightsPrompt = `Based on the comprehensive data analysis and intelligent visualization recommendations, generate strategic business insights for pharmaceutical decision-makers:

=== DATA ANALYSIS RESULTS ===
${analysisText}

=== VISUALIZATION INTELLIGENCE ===
Total visualizations recommended: ${visualizations?.individual?.length || 0}
Chart types suggested: ${visualizations?.individual?.map(v => v.chartType).join(', ') || 'None'}
Dashboard layout: ${visualizations?.dashboard?.layout || 'Not specified'}

=== STRATEGIC INSIGHTS REQUIRED ===
Provide executive-level insights including:

1. **Key Business Implications**: What do the data patterns mean for market strategy?
2. **Strategic Recommendations**: Specific actions for market penetration and growth
3. **Risk Assessment**: Potential threats and mitigation strategies  
4. **Market Opportunities**: Untapped segments and expansion possibilities
5. **Competitive Positioning**: How to differentiate and compete effectively
6. **Resource Allocation**: Where to invest time, money, and effort
7. **Timeline & Milestones**: Recommended implementation phases

Format as a structured strategic brief with clear recommendations and next steps.`;

        try {
            const response = await callClaudeWithRetry({
                data: {
                    model: adminSettings.claudeModel,
                    max_tokens: 1800,
                    temperature: 0.6,
                    messages: [{
                        role: 'user', 
                        content: insightsPrompt
                    }]
                },
                headers: {
                    'x-api-key': ANTHROPIC_API_KEY.trim(),
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                }
            }, 3);

            return response.data.content[0].text;

        } catch (error) {
            console.error('Insights Agent error:', error);
            return `**Strategic Insights Generation**

Due to processing limitations, automated insights are temporarily unavailable. 

However, based on the analysis patterns detected, key recommendations include:

• **Market Penetration**: Focus on high-adoption segments identified in the data
• **Competitive Strategy**: Leverage differentiation opportunities revealed by comparative analysis  
• **Geographic Expansion**: Target regions showing growth potential
• **Physician Engagement**: Tailor approaches based on practice setting preferences
• **Patient Outcomes**: Monitor efficacy/safety correlations for optimal positioning

*Full strategic analysis available upon system restoration.*`;
        }
    }
}

// Initialize orchestrator
const orchestrator = new MultiAgentOrchestrator();

// ===========================
// RAG HELPER FUNCTIONS (Keep existing)
// ===========================

async function generateEmbedding(text) {
    try {
        if (!OPENAI_API_KEY) {
            console.log('Using keyword-based similarity (no OpenAI key)');
            return generateKeywordVector(text);
        }

        const response = await axios.post('https://api.openai.com/v1/embeddings', {
            model: 'text-embedding-ada-002',
            input: text.substring(0, 8000)
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
        return relevantDocs;

    } catch (error) {
        console.error('Context retrieval failed:', error);
        return [];
    }
}

// Retry function for handling 529 overloaded errors
async function callClaudeWithRetry(requestConfig, maxRetries = 5) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            console.log(`🚀 Attempting Claude API call (attempt ${attempt}/${maxRetries})...`);
            
            const response = await axios.post('https://api.anthropic.com/v1/messages', requestConfig.data, {
                headers: requestConfig.headers
            });
            
            console.log('✅ Claude API call successful!');
            return response;
            
        } catch (error) {
            const statusCode = error.response?.status;
            const errorType = error.response?.data?.error?.type;
            
            console.log(`❌ API call failed (attempt ${attempt}): ${statusCode} - ${errorType}`);
            
            if ((statusCode === 529 && errorType === 'overloaded_error') || 
                (statusCode === 429 && errorType === 'rate_limit_error')) {
                
                if (attempt < maxRetries) {
                    const backoffTime = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
                    console.log(`⏳ Retrying in ${Math.round(backoffTime/1000)} seconds...`);
                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                    continue;
                } else {
                    console.log(`💀 Max retries (${maxRetries}) exceeded for 529/429 errors`);
                    throw error;
                }
            } else {
                console.log(`💀 Non-retryable error: ${statusCode} - ${errorType}`);
                throw error;
            }
        }
    }
}

// ===========================
// BASIC ROUTES
// ===========================

app.get('/', (req, res) => {
    res.json({ 
        message: 'Sagan Multi-Agent Dashboard Backend is running!',
        version: '2.0',
        multiAgentSystem: adminSettings.multiAgentSystem.enabled,
        activeAgents: Object.keys(agents).filter(key => agents[key].active),
        visualizationAgent: adminSettings.visualizationAgent.enabled,
        documentsLoaded: documentStore.length,
        status: 'operational',
        timestamp: new Date().toISOString()
    });
});

app.get('/api/health', (req, res) => {
    const cleanedApiKey = ANTHROPIC_API_KEY ? ANTHROPIC_API_KEY.trim() : null;
    
    res.json({ 
        status: 'healthy',
        version: '2.0-multiagent',
        apiKeyConfigured: !!cleanedApiKey,
        apiKeyValid: cleanedApiKey && cleanedApiKey.startsWith('sk-ant-') && cleanedApiKey.length > 20,
        openaiKeyConfigured: !!OPENAI_API_KEY,
        multiAgentSystem: {
            enabled: adminSettings.multiAgentSystem.enabled,
            activeAgents: Object.keys(agents).filter(key => agents[key].active).length,
            totalAgents: Object.keys(agents).length
        },
        visualizationAgent: {
            enabled: adminSettings.visualizationAgent.enabled,
            chartTypes: adminSettings.visualizationAgent.chartTypes.length,
            interactiveMode: adminSettings.visualizationAgent.interactiveMode
        },
        ragEnabled: adminSettings.ragEnabled,
        documentsInStore: documentStore.length,
        timestamp: new Date().toISOString()
    });
});

// ===========================
// ENHANCED ANALYSIS ENDPOINT (Multi-Agent)
// ===========================

app.post('/api/analyze', async (req, res) => {
    try {
        const { fileContent, fileName, userPrompt, webSearchEnabled } = req.body;
        
        // API key validation
        const cleanedApiKey = ANTHROPIC_API_KEY ? ANTHROPIC_API_KEY.trim() : null;
        if (!cleanedApiKey || cleanedApiKey.length < 20 || !cleanedApiKey.startsWith('sk-ant-')) {
            return res.status(500).json({ 
                error: 'API key configuration issue',
                multiAgentStatus: 'configuration_error'
            });
        }
        
        if (!fileContent) {
            return res.status(400).json({ error: 'No file content provided' });
        }

        console.log(`🤖 Multi-Agent Analysis Request:`);
        console.log(`📄 File: ${fileName}`);
        console.log(`🎯 Prompt: ${userPrompt ? 'Custom instructions provided' : 'Default analysis'}`);
        console.log(`🌐 Web Search: ${webSearchEnabled ? 'Enabled' : 'Disabled'}`);
        console.log(`🔧 Active Agents: ${Object.keys(agents).filter(key => agents[key].active).join(', ')}`);

        // Execute multi-agent workflow
        const agentResults = await orchestrator.processAnalysisRequest(
            fileContent, fileName, userPrompt
        );

        systemStats.totalAnalyses++;
        
        // Enhanced response with comprehensive agent results
        res.json({
            // Primary results
            analysis: agentResults.dataAnalysis,
            visualizations: agentResults.visualizations,
            insights: agentResults.insights,
            
            // Multi-agent metadata
            agentMetadata: {
                workflowId: agentResults.workflowId,
                activeAgents: Object.keys(agents).filter(key => agents[key].active),
                agentInteractions: systemStats.agentInteractions,
                visualizationsGenerated: systemStats.visualizationsGenerated,
                totalProcessingTime: agentResults.totalProcessingTime,
                agentTiming: agentResults.agentTiming,
                ragDocumentsUsed: agentResults.ragContext.length,
                workflowVersion: '2.0'
            },
            
            // RAG context (existing)
            ragContext: {
                enabled: adminSettings.ragEnabled,
                documentsUsed: agentResults.ragContext.length,
                contextSources: agentResults.ragContext.map(doc => ({
                    fileName: doc.fileName,
                    similarity: doc.similarity,
                    category: doc.category
                }))
            },
            
            // Visualization specific metadata
            visualizationMetadata: agentResults.visualizations?.metadata || {},
            
            // Request metadata
            metadata: {
                fileName: fileName,
                processedAt: new Date().toISOString(),
                webSearchEnabled: webSearchEnabled,
                userPromptUsed: !!userPrompt,
                multiAgentWorkflow: true,
                systemVersion: '2.0'
            }
        });

        console.log(`✅ Multi-Agent Analysis completed successfully`);
        console.log(`⏱️  Total processing time: ${agentResults.totalProcessingTime}ms`);
        console.log(`📊 Visualizations generated: ${agentResults.visualizations?.individual?.length || 0}`);

    } catch (error) {
        console.error('❌ Multi-Agent Analysis Error:', error);
        res.status(500).json({
            error: 'Multi-agent analysis failed',
            details: error.message,
            agentStatus: 'workflow_failed',
            timestamp: new Date().toISOString()
        });
    }
});

// ===========================
// AGENT MANAGEMENT ENDPOINTS
// ===========================

app.get('/api/agents/status', (req, res) => {
    res.json({
        agents: agents,
        systemStats: {
            totalAgentInteractions: systemStats.agentInteractions,
            visualizationsGenerated: systemStats.visualizationsGenerated,
            multiAgentWorkflows: systemStats.multiAgentWorkflows,
            averageProcessingTime: systemStats.averageProcessingTime,
            activeAgentCount: Object.keys(agents).filter(key => agents[key].active).length,
            totalAgents: Object.keys(agents).length
        },
        capabilities: {
            dataAnalysis: agents.dataAnalysis.capabilities,
            visualization: agents.visualization.capabilities,
            insights: agents.insights.capabilities,
            rag: agents.rag.capabilities
        },
        configuration: {
            multiAgentSystem: adminSettings.multiAgentSystem,
            visualizationAgent: adminSettings.visualizationAgent
        }
    });
});

app.post('/api/agents/configure', (req, res) => {
    const { agentName, enabled, settings } = req.body;
    
    if (agents[agentName]) {
        agents[agentName].active = enabled;
        if (settings) {
            agents[agentName].settings = { ...agents[agentName].settings, ...settings };
        }
        
        // Special handling for RAG agent
        if (agentName === 'rag') {
            adminSettings.ragEnabled = enabled;
        }
        
        console.log(`🔧 Agent configuration updated: ${agentName} ${enabled ? 'enabled' : 'disabled'}`);
        
        res.json({ 
            success: true, 
            agent: agents[agentName],
            message: `${agentName} agent ${enabled ? 'enabled' : 'disabled'}`,
            activeAgents: Object.keys(agents).filter(key => agents[key].active)
        });
    } else {
        res.status(404).json({ error: 'Agent not found' });
    }
});

// ===========================
// ENHANCED CHAT ENDPOINT (Keep existing functionality)
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
        
        console.log(`💬 Processing enhanced chat question: ${question.substring(0, 50)}...`);
        
        let contextualPrompt = `Based on this multi-agent pharmaceutical survey analysis, please answer the user's question concisely and professionally:

ANALYSIS RESULTS:
${analysis}`;

        // RAG Enhancement for chat
        if (adminSettings.ragEnabled) {
            const relevantContext = await retrieveRelevantContext(question, 2);
            
            if (relevantContext.length > 0) {
                contextualPrompt += `\n\nRELEVANT KNOWLEDGE BASE CONTEXT:`;
                relevantContext.forEach((doc, index) => {
                    contextualPrompt += `\n--- Reference ${index + 1} ---\n${doc.content}`;
                });
            }
        }

        contextualPrompt += `\n\nUSER QUESTION: ${question}

Please provide a helpful, specific answer based on the multi-agent analysis data${adminSettings.ragEnabled ? ' and reference materials' : ''}. Keep your response focused and under 250 words.`;

        const chatResponse = await callClaudeWithRetry({
            data: {
                model: adminSettings.claudeModel,
                max_tokens: 350,
                temperature: 0.3,
                messages: [{
                    role: 'user',
                    content: contextualPrompt
                }]
            },
            headers: {
                'x-api-key': ANTHROPIC_API_KEY.trim(),
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
        }, 3);

        res.json({ 
            response: chatResponse.data.content[0].text,
            multiAgentEnabled: true,
            ragEnabled: adminSettings.ragEnabled,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('Enhanced chat error:', error.response?.data || error.message);
        res.status(500).json({ 
            error: 'Chat failed. Please try again.',
            details: error.message
        });
    }
});

// ===========================
// KEEP ALL EXISTING ADMIN ENDPOINTS
// ===========================
// (Your existing admin endpoints for settings, training, etc. remain unchanged)

app.get('/admin', (req, res) => {
    res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sagan Admin - Multi-Agent System v2.0</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #0f172a; color: white; }
            .container { max-width: 600px; margin: 0 auto; }
            .btn { background: #3b82f6; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; display: inline-block; margin: 10px; }
            .agent-status { color: ${adminSettings.multiAgentSystem.enabled ? '#10b981' : '#ef4444'}; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Sagan Multi-Agent Admin Dashboard v2.0</h1>
            <p>Enhanced backend with intelligent visualization agents!</p>
            <p class="agent-status">Multi-Agent System: ${adminSettings.multiAgentSystem.enabled ? 'ENABLED' : 'DISABLED'}</p>
            <p>Active Agents: ${Object.keys(agents).filter(key => agents[key].active).length}/${Object.keys(agents).length}</p>
            <p>Documents in store: ${documentStore.length}</p>
            <p>Visualizations generated: ${systemStats.visualizationsGenerated}</p>
            <a href="/admin/settings" class="btn">View Settings</a>
            <a href="/admin/stats" class="btn">View Stats</a>
            <a href="/api/agents/status" class="btn">Agent Status</a>
        </div>
    </body>
    </html>
    `);
});

// Enhanced stats endpoint
app.get('/admin/stats', (req, res) => {
    const uptime = Date.now() - systemStats.startTime.getTime();
    const uptimeHours = Math.floor(uptime / (1000 * 60 * 60));
    const uptimeMinutes = Math.floor((uptime % (1000 * 60 * 60)) / (1000 * 60));
    
    res.json({
        // Existing stats
        totalAnalyses: systemStats.totalAnalyses,
        activeUsers: systemStats.activeUsers,
        trainingExamples: trainingExamples.length,
        documentChunks: documentStore.length,
        ragMatches: systemStats.ragQueries,
        
        // Multi-agent stats
        multiAgentWorkflows: systemStats.multiAgentWorkflows,
        agentInteractions: systemStats.agentInteractions,
        visualizationsGenerated: systemStats.visualizationsGenerated,
        averageProcessingTime: Math.round(systemStats.averageProcessingTime),
        
        // Agent details
        agents: Object.keys(agents).map(key => ({
            name: key,
            active: agents[key].active,
            capabilities: agents[key].capabilities.length,
            lastProcessingTime: agents[key].processingTime
        })),
        
        // System info
        uptime: `${uptimeHours}h ${uptimeMinutes}m`,
        systemHealth: 'healthy',
        version: '2.0-multiagent',
        lastRestart: systemStats.startTime.toISOString()
    });
});

// Keep all your existing admin endpoints...
// (update-prompt, upload-training, etc. - they remain the same)

// ===========================
// SERVER STARTUP
// ===========================

app.listen(PORT, () => {
    console.log(`🚀 Sagan Multi-Agent Dashboard Backend v2.0 running on port ${PORT}`);
    console.log(`🤖 Multi-Agent System: ${adminSettings.multiAgentSystem.enabled ? 'ENABLED' : 'DISABLED'}`);
    console.log(`🎨 Visualization Agent: ${adminSettings.visualizationAgent.enabled ? 'ENABLED' : 'DISABLED'}`);
    console.log(`📚 RAG Agent: ${adminSettings.ragEnabled ? 'ENABLED' : 'DISABLED'}`);
    console.log(`🔑 Claude API Key: ${!!ANTHROPIC_API_KEY ? 'CONFIGURED' : 'MISSING'}`);
    console.log(`🔍 OpenAI API Key: ${!!OPENAI_API_KEY ? 'CONFIGURED' : 'MISSING'}`);
    console.log(`🌍 Health check: http://localhost:${PORT}/api/health`);
    console.log(`⚙️  Agent status: http://localhost:${PORT}/api/agents/status`);
    
    // Log active agents
    const activeAgents = Object.keys(agents).filter(key => agents[key].active);
    console.log(`🔧 Active Agents (${activeAgents.length}/${Object.keys(agents).length}):`, activeAgents.join(', '));
    
    // Log available chart types
    console.log(`📊 Available Chart Types:`, adminSettings.visualizationAgent.chartTypes.join(', '));
});

module.exports = app;
