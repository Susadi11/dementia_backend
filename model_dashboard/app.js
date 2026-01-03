// Model Dashboard App
class ModelDashboard {
    constructor() {
        this.models = [];
        this.currentFilter = 'all';
        this.init();
    }

    async init() {
        await this.loadModels();
        this.setupEventListeners();
        this.renderModels();
        this.updateStats();
    }

    async loadModels() {
        try {
            // Try to load from models_registry.json
            // First try current directory (if copied here)
            let response;
            try {
                response = await fetch('models_registry.json');
                if (!response.ok) throw new Error('Not found in current directory');
            } catch {
                // If not found, try parent directory
                response = await fetch('../models_registry.json');
            }

            const data = await response.json();
            this.models = data.models || [];
            this.registryData = data;
        } catch (error) {
            console.error('Error loading models:', error);
            this.showError('Failed to load models data. Make sure to run: python3 -m http.server 8000 -d model_dashboard');
        }
    }

    setupEventListeners() {
        // Filter buttons
        const filterButtons = document.querySelectorAll('.filter-btn');
        filterButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                filterButtons.forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.currentFilter = e.target.dataset.category;
                this.renderModels();
            });
        });

        // Modal close
        const closeBtn = document.getElementById('closeModal');
        const modal = document.getElementById('modelModal');
        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
        });

        // Close modal on outside click
        window.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    }

    filterModels() {
        if (this.currentFilter === 'all') {
            return this.models;
        }
        return this.models.filter(model => model.category === this.currentFilter);
    }

    renderModels() {
        const grid = document.getElementById('modelsGrid');
        const filteredModels = this.filterModels();

        if (filteredModels.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <div class="empty-state-text">No models found</div>
                </div>
            `;
            return;
        }

        grid.innerHTML = filteredModels.map(model => this.createModelCard(model)).join('');

        // Add click listeners to cards
        document.querySelectorAll('.model-card').forEach((card, index) => {
            card.addEventListener('click', () => {
                this.showModelDetails(filteredModels[index]);
            });
        });
    }

    createModelCard(model) {
        // Get best accuracy/score
        let accuracyBadge = '';
        if (model.metrics && model.metrics.best_score !== undefined) {
            const score = (model.metrics.best_score * 100).toFixed(1);
            accuracyBadge = `
                <div class="meta-item">
                    <span class="meta-label">Best Score</span>
                    <span class="accuracy-badge">${score}%</span>
                </div>
            `;
        }

        // Format training date
        let trainingDate = 'Unknown';
        if (model.training_date && model.training_date !== 'Unknown') {
            trainingDate = new Date(model.training_date).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        }

        // Get dataset size
        let datasetSize = 'Unknown';
        if (model.dataset_info && model.dataset_info.total_samples !== 'Unknown') {
            datasetSize = model.dataset_info.total_samples.toLocaleString();
        }

        return `
            <div class="model-card" data-id="${model.id}">
                <div class="model-header">
                    <div class="model-category">${model.category}</div>
                    <h3 class="model-name">${model.name}</h3>
                    <p class="model-type">${model.type} • ${model.algorithm}</p>
                </div>
                <p class="model-purpose">${model.purpose}</p>
                <div class="model-meta">
                    ${accuracyBadge}
                    <div class="meta-item">
                        <span class="meta-label">Training Date</span>
                        <span class="meta-value">${trainingDate}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Dataset Size</span>
                        <span class="meta-value">${datasetSize}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">File Size</span>
                        <span class="meta-value">${model.file_size_mb} MB</span>
                    </div>
                </div>
            </div>
        `;
    }

    showModelDetails(model) {
        const modal = document.getElementById('modelModal');
        const modalBody = document.getElementById('modalBody');

        modalBody.innerHTML = this.createModelDetailView(model);
        modal.style.display = 'block';
    }

    createModelDetailView(model) {
        let html = `
            <h2 class="modal-title">${model.name}</h2>
            <p class="modal-subtitle">${model.purpose}</p>

            <div class="detail-section">
                <h3 class="section-title">Model Information</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-label">Model Type</div>
                        <div class="detail-value">${model.type}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Algorithm</div>
                        <div class="detail-value">${model.algorithm}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Category</div>
                        <div class="detail-value">${model.category}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">File Size</div>
                        <div class="detail-value">${model.file_size_mb} MB</div>
                    </div>
                </div>
            </div>
        `;

        // Dataset Information
        if (model.dataset_info) {
            html += `
                <div class="detail-section">
                    <h3 class="section-title">Dataset Information</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-label">Total Samples</div>
                            <div class="detail-value">${(model.dataset_info.total_samples || 'Unknown').toLocaleString()}</div>
                        </div>
                        ${model.dataset_info.synthetic_samples ? `
                        <div class="detail-item">
                            <div class="detail-label">Synthetic Samples</div>
                            <div class="detail-value">${model.dataset_info.synthetic_samples.toLocaleString()}</div>
                        </div>
                        ` : ''}
                        ${model.dataset_info.pitt_samples ? `
                        <div class="detail-item">
                            <div class="detail-label">PITT Corpus Samples</div>
                            <div class="detail-value">${model.dataset_info.pitt_samples.toLocaleString()}</div>
                        </div>
                        ` : ''}
                        ${model.metrics && model.metrics.feature_count ? `
                        <div class="detail-item">
                            <div class="detail-label">Features</div>
                            <div class="detail-value">${model.metrics.feature_count}</div>
                        </div>
                        ` : ''}
                    </div>
                    ${model.dataset_info.data_sources && model.dataset_info.data_sources.length > 0 ? `
                    <div style="margin-top: 16px;">
                        <div class="detail-label" style="margin-bottom: 8px;">Data Sources</div>
                        <ul class="feature-list">
                            ${model.dataset_info.data_sources.map(source => `<li>${source}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                </div>
            `;
        }

        // Metrics / Performance
        if (model.metrics && model.metrics.algorithm_results) {
            html += `
                <div class="detail-section">
                    <h3 class="section-title">Model Performance</h3>
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>Algorithm</th>
                                <th>CV Score (Mean)</th>
                                <th>CV Score (Std)</th>
                                <th>Test Score</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            for (const [algo, scores] of Object.entries(model.metrics.algorithm_results)) {
                const algoName = algo.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                const cvMean = (scores.cv_score_mean * 100).toFixed(2);
                const cvStd = (scores.cv_score_std * 100).toFixed(3);
                const testScore = (scores.test_score * 100).toFixed(2);

                html += `
                    <tr>
                        <td><strong>${algoName}</strong></td>
                        <td>
                            ${cvMean}%
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${cvMean}%"></div>
                            </div>
                        </td>
                        <td>${cvStd}%</td>
                        <td>
                            ${testScore}%
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${testScore}%"></div>
                            </div>
                        </td>
                    </tr>
                `;
            }

            html += `
                        </tbody>
                    </table>
                </div>
            `;
        }

        // Special handling for LLaMA model
        if (model.id === 'llama_3_2_3b_dementia_care') {
            html += `
                <div class="detail-section">
                    <h3 class="section-title">LLaMA Configuration</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-label">Base Model</div>
                            <div class="detail-value">${model.base_model}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">LoRA Adapter</div>
                            <div class="detail-value">${model.lora_adapter}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Training Dataset</div>
                            <div class="detail-value">${model.training_dataset}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Training Method</div>
                            <div class="detail-value">${model.training_method}</div>
                        </div>
                    </div>
                    ${model.features ? `
                    <div style="margin-top: 16px;">
                        <div class="detail-label" style="margin-bottom: 8px;">Features</div>
                        <ul class="feature-list">
                            ${model.features.map(feature => `<li>${feature}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                    ${model.inference_hardware ? `
                    <div style="margin-top: 16px;">
                        <div class="detail-label" style="margin-bottom: 8px;">Inference Hardware</div>
                        <ul class="feature-list">
                            ${model.inference_hardware.map(hw => `<li>${hw}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                </div>
            `;

            // LLM-specific metrics display
            if (model.metrics) {
                if (model.metrics.model_info) {
                    html += `
                        <div class="detail-section">
                            <h3 class="section-title">Model Information</h3>
                            <div class="detail-grid">
                                ${model.metrics.model_info.parameters ? `
                                <div class="detail-item">
                                    <div class="detail-label">Parameters</div>
                                    <div class="detail-value">${model.metrics.model_info.parameters}</div>
                                </div>
                                ` : ''}
                                ${model.metrics.model_info.fine_tuning_method ? `
                                <div class="detail-item">
                                    <div class="detail-label">Fine-tuning Method</div>
                                    <div class="detail-value">${model.metrics.model_info.fine_tuning_method}</div>
                                </div>
                                ` : ''}
                                ${model.metrics.model_info.quantization ? `
                                <div class="detail-item">
                                    <div class="detail-label">Quantization</div>
                                    <div class="detail-value">${model.metrics.model_info.quantization}</div>
                                </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
                }

                if (model.metrics.training_info) {
                    html += `
                        <div class="detail-section">
                            <h3 class="section-title">Training Information</h3>
                            <div class="detail-grid">
                                ${model.metrics.training_info.base_dataset ? `
                                <div class="detail-item">
                                    <div class="detail-label">Base Dataset</div>
                                    <div class="detail-value">${model.metrics.training_info.base_dataset}</div>
                                </div>
                                ` : ''}
                                ${model.metrics.training_info.domain_adaptation ? `
                                <div class="detail-item">
                                    <div class="detail-label">Domain Adaptation</div>
                                    <div class="detail-value">${model.metrics.training_info.domain_adaptation}</div>
                                </div>
                                ` : ''}
                                ${model.metrics.training_info.training_samples ? `
                                <div class="detail-item">
                                    <div class="detail-label">Training Samples</div>
                                    <div class="detail-value">${model.metrics.training_info.training_samples}</div>
                                </div>
                                ` : ''}
                            </div>
                            ${model.metrics.training_info.note ? `
                            <div style="margin-top: 16px; padding: 16px; background: var(--primary-bg); border-radius: var(--radius-sm);">
                                <div class="detail-label" style="margin-bottom: 8px;">Note</div>
                                <p style="color: var(--text-primary); font-size: 14px; line-height: 1.6;">${model.metrics.training_info.note}</p>
                            </div>
                            ` : ''}
                        </div>
                    `;
                }

                if (model.metrics.training_metrics) {
                    html += `
                        <div class="detail-section">
                            <h3 class="section-title">Training Metrics</h3>
                            <div class="detail-grid">
                                ${model.metrics.training_metrics.final_perplexity ? `
                                <div class="detail-item" style="grid-column: 1 / -1;">
                                    <div class="detail-label">Perplexity (Primary LLM Metric)</div>
                                    <div class="detail-value" style="font-size: 24px; color: var(--success-green);">${model.metrics.training_metrics.final_perplexity.toFixed(2)}</div>
                                    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">
                                        ${model.metrics.training_metrics.initial_perplexity ? `Reduced from ${model.metrics.training_metrics.initial_perplexity.toFixed(0)} (${model.metrics.training_metrics.perplexity_reduction} improvement)` : ''}
                                    </div>
                                </div>
                                ` : ''}
                                ${model.metrics.training_metrics.initial_loss ? `
                                <div class="detail-item">
                                    <div class="detail-label">Initial Loss</div>
                                    <div class="detail-value">${model.metrics.training_metrics.initial_loss.toFixed(4)}</div>
                                </div>
                                ` : ''}
                                ${model.metrics.training_metrics.final_loss ? `
                                <div class="detail-item">
                                    <div class="detail-label">Final Loss</div>
                                    <div class="detail-value">${model.metrics.training_metrics.final_loss.toFixed(4)}</div>
                                </div>
                                ` : ''}
                                ${model.metrics.training_metrics.loss_reduction ? `
                                <div class="detail-item">
                                    <div class="detail-label">Loss Reduction</div>
                                    <div class="detail-value" style="color: var(--success-green);">${model.metrics.training_metrics.loss_reduction}</div>
                                </div>
                                ` : ''}
                            </div>
                            ${model.metrics.training_metrics.convergence ? `
                            <div style="margin-top: 16px; padding: 16px; background: linear-gradient(135deg, rgba(52, 199, 89, 0.1), rgba(48, 209, 88, 0.05)); border-radius: var(--radius-sm); border-left: 3px solid var(--success-green);">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="font-size: 20px;">🌟</span>
                                    <p style="color: var(--text-primary); font-size: 14px; font-weight: 500; margin: 0;">${model.metrics.training_metrics.convergence}</p>
                                </div>
                            </div>
                            ` : ''}
                        </div>
                    `;
                }

                if (model.metrics.evaluation_metrics) {
                    html += `
                        <div class="detail-section">
                            <h3 class="section-title">Evaluation Metrics</h3>
                            <ul class="feature-list">
                    `;
                    for (const [key, value] of Object.entries(model.metrics.evaluation_metrics)) {
                        const label = key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                        html += `<li><strong>${label}:</strong> ${value}</li>`;
                    }
                    html += `
                            </ul>
                        </div>
                    `;
                }
            }

            // HuggingFace link
            if (model.huggingface_url) {
                html += `
                    <div class="detail-section">
                        <a href="${model.huggingface_url}" target="_blank" class="link-btn">
                            View on HuggingFace →
                        </a>
                    </div>
                `;
            }
        }

        // Features used (for text models)
        if (model.features_used && model.features_used.length > 0) {
            html += `
                <div class="detail-section">
                    <h3 class="section-title">Features Used</h3>
                    <ul class="feature-list">
                        ${model.features_used.map(feature => `<li>${feature}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // File Information
        html += `
            <div class="detail-section">
                <h3 class="section-title">File Information</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-label">File Path</div>
                        <div class="detail-value" style="font-size: 12px; word-break: break-all;">${model.file_path}</div>
                    </div>
                    ${model.has_scaler !== undefined ? `
                    <div class="detail-item">
                        <div class="detail-label">Has Scaler</div>
                        <div class="detail-value">${model.has_scaler ? '✓ Yes' : '✗ No'}</div>
                    </div>
                    ` : ''}
                    <div class="detail-item">
                        <div class="detail-label">Training Date</div>
                        <div class="detail-value">
                            ${model.training_date !== 'Unknown'
                                ? new Date(model.training_date).toLocaleDateString('en-US', {
                                    year: 'numeric',
                                    month: 'long',
                                    day: 'numeric',
                                    hour: '2-digit',
                                    minute: '2-digit'
                                })
                                : 'Unknown'
                            }
                        </div>
                    </div>
                </div>
            </div>
        `;

        return html;
    }

    updateStats() {
        const totalModels = document.getElementById('totalModels');
        const lastUpdated = document.getElementById('lastUpdated');

        totalModels.textContent = this.models.length;

        if (this.registryData && this.registryData.last_updated) {
            const date = new Date(this.registryData.last_updated);
            lastUpdated.textContent = date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
            });
        } else {
            lastUpdated.textContent = 'Unknown';
        }
    }

    showError(message) {
        const grid = document.getElementById('modelsGrid');
        grid.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">⚠️</div>
                <div class="empty-state-text">${message}</div>
            </div>
        `;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ModelDashboard();
});
