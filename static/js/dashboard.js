/**
 * Crisis Sentinel Dashboard JavaScript
 * Handles data fetching, visualization, and user interactions
 */

// Dashboard state
const dashboardState = {
    currentStatus: null,
    lastUpdated: null,
    sentimentChart: null,
    marketChart: null,
    isLoading: false
};

// DOM Elements
const elements = {
    alertLevel: document.getElementById('alert-level-number'),
    alertLevelText: document.getElementById('alert-level-text'),
    alertLevelIndicator: document.getElementById('alert-level-indicator'),
    alertDescription: document.getElementById('alert-description'),
    keyInsightsList: document.getElementById('key-insights-list'),
    sentimentSummary: document.getElementById('sentiment-summary'),
    sentimentIndicators: document.getElementById('sentiment-indicators'),
    marketSummary: document.getElementById('market-summary'),
    marketIndicators: document.getElementById('market-indicators'),
    lastUpdated: document.getElementById('last-updated'),
    refreshBtn: document.getElementById('refresh-btn'),
    reportType: document.getElementById('report-type'),
    timeHorizon: document.getElementById('time-horizon'),
    generateReportBtn: document.getElementById('generate-report-btn'),
    reportContent: document.getElementById('report-content')
};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
    setupEventListeners();
});

/**
 * Initialize the dashboard
 */
function initializeDashboard() {
    // Initialize charts
    initializeCharts();
    
    // Fetch initial data
    fetchCurrentStatus();
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Refresh button
    elements.refreshBtn.addEventListener('click', () => {
        if (!dashboardState.isLoading) {
            fetchCurrentStatus();
        }
    });
    
    // Generate report button
    elements.generateReportBtn.addEventListener('click', generateReport);
}

/**
 * Initialize chart visualizations
 */
function initializeCharts() {
    // Sentiment chart
    const sentimentCtx = document.getElementById('sentiment-chart').getContext('2d');
    dashboardState.sentimentChart = new Chart(sentimentCtx, {
        type: 'line',
        data: {
            labels: ['7 Days Ago', '6 Days Ago', '5 Days Ago', '4 Days Ago', '3 Days Ago', '2 Days Ago', 'Today'],
            datasets: [{
                label: 'Sentiment Score',
                data: [0, 0, 0, 0, 0, 0, 0],
                borderColor: '#007bff',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    suggestedMin: -1,
                    suggestedMax: 1
                }
            }
        }
    });
    
    // Market chart
    const marketCtx = document.getElementById('market-chart').getContext('2d');
    dashboardState.marketChart = new Chart(marketCtx, {
        type: 'bar',
        data: {
            labels: ['Yield Curve', 'Volatility', 'Credit Spreads', 'Liquidity', 'Market Indices'],
            datasets: [{
                label: 'Risk Level',
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.7)',  // Green
                    'rgba(255, 193, 7, 0.7)',  // Yellow
                    'rgba(253, 126, 20, 0.7)', // Orange
                    'rgba(220, 53, 69, 0.7)',  // Red
                    'rgba(108, 117, 125, 0.7)' // Gray
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(255, 193, 7)',
                    'rgb(253, 126, 20)',
                    'rgb(220, 53, 69)',
                    'rgb(108, 117, 125)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 5,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

/**
 * Fetch current crisis status from API
 */
function fetchCurrentStatus() {
    setLoading(true);
    
    fetch('/api/current_status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            dashboardState.currentStatus = data;
            dashboardState.lastUpdated = new Date();
            updateDashboard(data);
            setLoading(false);
        })
        .catch(error => {
            console.error('Error fetching current status:', error);
            showError('Failed to fetch current status. Please try again later.');
            setLoading(false);
        });
}

/**
 * Update dashboard with new data
 * @param {Object} data - Current status data
 */
function updateDashboard(data) {
    if (!data) return;
    
    // Update alert level
    updateAlertLevel(data.defcon_level);
    
    // Update insights
    updateInsights(data.insights);
    
    // Update sentiment section
    updateSentimentSection(data.sentiment_summary);
    
    // Update market section
    updateMarketSection(data.market_summary);
    
    // Update last updated timestamp
    updateTimestamp();
}

/**
 * Update the alert level display
 * @param {number} level - Alert level (1-5)
 */
function updateAlertLevel(level) {
    // Clear previous classes
    elements.alertLevelIndicator.className = 'alert-level-indicator';
    
    // Add new class based on level
    elements.alertLevelIndicator.classList.add(`alert-level-${level}`);
    
    // Update number and text
    elements.alertLevel.textContent = level;
    
    // Update level text
    const levelTexts = [
        'Unknown',
        'Very Low',
        'Low',
        'Moderate',
        'High',
        'Very High'
    ];
    
    elements.alertLevelText.textContent = levelTexts[level] || 'Unknown';
    
    // Update description
    const descriptions = [
        'Alert level unknown. System may be initializing.',
        'Very low risk of crisis. Normal market conditions.',
        'Low risk of crisis. Minor market stress indicators.',
        'Moderate risk of crisis. Several concerning indicators detected.',
        'High risk of crisis. Multiple warning signs present.',
        'Very high risk of crisis. Immediate attention required.'
    ];
    
    elements.alertDescription.innerHTML = `<p>${descriptions[level] || descriptions[0]}</p>`;
}

/**
 * Update insights list
 * @param {Array} insights - List of insight objects
 */
function updateInsights(insights) {
    if (!insights || !insights.length) {
        elements.keyInsightsList.innerHTML = '<li>No insights available</li>';
        return;
    }
    
    // Clear previous insights
    elements.keyInsightsList.innerHTML = '';
    
    // Add new insights (limit to top 5)
    insights.slice(0, 5).forEach(insight => {
        const li = document.createElement('li');
        
        // Add severity indicator
        const severitySpan = document.createElement('span');
        severitySpan.className = `severity-indicator severity-${insight.severity || 0}`;
        li.appendChild(severitySpan);
        
        // Add description
        li.appendChild(document.createTextNode(insight.description));
        
        elements.keyInsightsList.appendChild(li);
    });
}

/**
 * Update sentiment analysis section
 * @param {Object} sentimentData - Sentiment summary data
 */
function updateSentimentSection(sentimentData) {
    if (!sentimentData) {
        elements.sentimentSummary.innerHTML = '<p>No sentiment data available</p>';
        elements.sentimentIndicators.innerHTML = '<li>No indicators available</li>';
        return;
    }
    
    // Update summary
    elements.sentimentSummary.innerHTML = `<p>${sentimentData.text || 'No sentiment summary available'}</p>`;
    
    // Update indicators
    elements.sentimentIndicators.innerHTML = '';
    
    // Add direction
    const directionItem = document.createElement('li');
    directionItem.innerHTML = `<strong>Direction:</strong> ${sentimentData.sentiment_direction || 'Unknown'}`;
    elements.sentimentIndicators.appendChild(directionItem);
    
    // Add shift
    const shiftItem = document.createElement('li');
    shiftItem.innerHTML = `<strong>Trend:</strong> ${sentimentData.shift_description || 'Unknown'}`;
    elements.sentimentIndicators.appendChild(shiftItem);
    
    // Add risk level
    const riskItem = document.createElement('li');
    riskItem.innerHTML = `<strong>Risk Level:</strong> ${sentimentData.risk_description || 'Unknown'}`;
    elements.sentimentIndicators.appendChild(riskItem);
    
    // Add concerns
    if (sentimentData.concerns && sentimentData.concerns.length) {
        const concernsItem = document.createElement('li');
        concernsItem.innerHTML = `<strong>Key Concerns:</strong>`;
        
        const concernsList = document.createElement('ul');
        sentimentData.concerns.forEach(concern => {
            const concernLi = document.createElement('li');
            concernLi.textContent = concern;
            concernsList.appendChild(concernLi);
        });
        
        concernsItem.appendChild(concernsList);
        elements.sentimentIndicators.appendChild(concernsItem);
    }
    
    // Update chart with dummy data for now
    // In a real implementation, this would use historical sentiment data
    updateSentimentChart();
}

/**
 * Update market indicators section
 * @param {Object} marketData - Market summary data
 */
function updateMarketSection(marketData) {
    if (!marketData) {
        elements.marketSummary.innerHTML = '<p>No market data available</p>';
        elements.marketIndicators.innerHTML = '<li>No indicators available</li>';
        return;
    }
    
    // Update summary
    elements.marketSummary.innerHTML = `<p>${marketData.text || 'No market summary available'}</p>`;
    
    // Update indicators
    elements.marketIndicators.innerHTML = '';
    
    // Add risk level
    const riskItem = document.createElement('li');
    riskItem.innerHTML = `<strong>Risk Level:</strong> ${marketData.risk_description || 'Unknown'}`;
    elements.marketIndicators.appendChild(riskItem);
    
    // Add key findings
    if (marketData.key_findings && marketData.key_findings.length) {
        const findingsItem = document.createElement('li');
        findingsItem.innerHTML = `<strong>Key Findings:</strong>`;
        
        const findingsList = document.createElement('ul');
        marketData.key_findings.forEach(finding => {
            const findingLi = document.createElement('li');
            findingLi.textContent = finding;
            findingsList.appendChild(findingLi);
        });
        
        findingsItem.appendChild(findingsList);
        elements.marketIndicators.appendChild(findingsItem);
    }
    
    // Update chart with dummy data for now
    // In a real implementation, this would use actual market indicator data
    updateMarketChart();
}

/**
 * Update sentiment chart with data
 */
function updateSentimentChart() {
    // In a real implementation, this would use historical sentiment data
    // For now, we'll use random data as a placeholder
    const randomData = Array.from({length: 7}, () => Math.random() * 2 - 1);
    
    dashboardState.sentimentChart.data.datasets[0].data = randomData;
    dashboardState.sentimentChart.update();
}

/**
 * Update market chart with data
 */
function updateMarketChart() {
    // In a real implementation, this would use actual market indicator data
    // For now, we'll use random data as a placeholder
    const randomData = Array.from({length: 5}, () => Math.floor(Math.random() * 5) + 1);
    
    dashboardState.marketChart.data.datasets[0].data = randomData;
    dashboardState.marketChart.update();
}

/**
 * Update last updated timestamp
 */
function updateTimestamp() {
    if (dashboardState.lastUpdated) {
        const formattedTime = dashboardState.lastUpdated.toLocaleTimeString();
        elements.lastUpdated.textContent = `Last updated: ${formattedTime}`;
    }
}

/**
 * Generate a crisis prediction report
 */
function generateReport() {
    const reportType = elements.reportType.value;
    const timeHorizon = elements.timeHorizon.value;
    
    // Show loading state
    elements.reportContent.innerHTML = `
        <div class="text-center p-4">
            <div class="loading-spinner mb-3"></div>
            <p>Generating ${reportType} report for ${timeHorizon} term...</p>
        </div>
    `;
    
    // Make API request
    fetch('/api/generate_report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            report_type: reportType,
            time_horizon: timeHorizon
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'success' && data.report) {
            displayReport(data.report);
        } else {
            showError('Failed to generate report. Please try again later.');
        }
    })
    .catch(error => {
        console.error('Error generating report:', error);
        showError('Failed to generate report. Please try again later.');
    });
}

/**
 * Display a generated report
 * @param {Object} report - Report data
 */
function displayReport(report) {
    // Format the report for display
    let reportHtml = `
        <div class="report-header mb-4">
            <h3>${report.title}</h3>
            <p class="text-muted">Generated on ${new Date(report.generated_at).toLocaleString()}</p>
            <div class="alert alert-${getAlertClassForLevel(report.alert_level)}">
                Current Crisis Alert Level: ${report.alert_level}/5
            </div>
        </div>
    `;
    
    // Add report sections
    if (report.sections && report.sections.length) {
        report.sections.forEach(section => {
            reportHtml += `
                <div class="report-section">
                    <h4>${section.title}</h4>
                    <p>${section.content}</p>
                </div>
            `;
        });
    }
    
    // Update report content
    elements.reportContent.innerHTML = reportHtml;
}

/**
 * Set loading state
 * @param {boolean} isLoading - Whether the dashboard is loading
 */
function setLoading(isLoading) {
    dashboardState.isLoading = isLoading;
    
    if (isLoading) {
        elements.refreshBtn.querySelector('i').classList.add('refresh-spin');
        elements.refreshBtn.disabled = true;
    } else {
        elements.refreshBtn.querySelector('i').classList.remove('refresh-spin');
        elements.refreshBtn.disabled = false;
    }
}

/**
 * Show an error message
 * @param {string} message - Error message to display
 */
function showError(message) {
    // You could implement a toast notification system here
    console.error(message);
    
    // For now, just show an alert
    alert(message);
}

/**
 * Get Bootstrap alert class for a given alert level
 * @param {number} level - Alert level (1-5)
 * @returns {string} - Bootstrap alert class
 */
function getAlertClassForLevel(level) {
    const alertClasses = [
        'secondary', // Unknown
        'success',   // Very Low
        'info',      // Low
        'warning',   // Moderate
        'warning',   // High
        'danger'     // Very High
    ];
    
    return alertClasses[level] || alertClasses[0];
} 