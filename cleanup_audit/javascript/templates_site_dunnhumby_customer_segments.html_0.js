
document.addEventListener('DOMContentLoaded', function() {
    const csrftoken = (document.cookie.split('; ').find(r => r.startsWith('csrftoken=')) || '').split('=')[1] || '';
    const detailModalEl = document.getElementById('detailModal');
    if (!detailModalEl) return;

    // -----------------------------------------------------------------
    // ۱. تابع نمایش مودال (بدون تغییر)
    // -----------------------------------------------------------------
    function showDetailModal(title, body, footer = null) {
        const modalTitle = detailModalEl.querySelector('.modal-title');
        const modalBody = detailModalEl.querySelector('.modal-body');
        const modalFooter = detailModalEl.querySelector('.modal-footer');

        if (modalTitle) modalTitle.innerHTML = title;
        if (modalBody) modalBody.innerHTML = body;
        if (modalFooter) {
            if (footer) {
                modalFooter.innerHTML = footer;
            } else {
                modalFooter.innerHTML = '<button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>';
            }
        }
        
        const modal = bootstrap.Modal.getOrCreateInstance(detailModalEl);
        modal.show();
    }

    // -----------------------------------------------------------------
    // ۲. تابع مرکزی برای دریافت و نمایش داده‌های هر دو نوع مودال
    // -----------------------------------------------------------------
    async function fetchAndDisplayPage(type, identifier, page) {
        const form = new FormData();
        form.append('csrfmiddlewaretoken', csrftoken);
        form.append('page', page);

        let apiUrl = '';
        let titlePrefix = '';
        
        if (type === 'rfm') {
            form.append('rfm_segment', identifier);
            apiUrl = '/analysis/api/segment/';
            titlePrefix = 'Segment: ';
        } else if (type === 'churn') {
            form.append('churn_risk', identifier);
            apiUrl = '/analysis/api/churn/';
            titlePrefix = 'Churn Risk: ';
        }

        const res = await fetch(apiUrl, { method: 'POST', body: form });
        const j = await res.json();
        
        if (j.error) {
            return showDetailModal(titlePrefix + identifier, `<div class="text-danger">${j.error}</div>`);
        }

        const m = j.metrics || {};
        const p = j.pagination || {};
        let bodyHtml = '';

        // ساخت بخش متریک‌ها بر اساس نوع مودال
        if (type === 'rfm') {
            bodyHtml = `<div class="row g-3">
                <div class="col-3 text-center"><div class="text-muted small">Customers</div><div class="fw-bold">${m.customers || 0}</div></div>
                <div class="col-3 text-center"><div class="text-muted small">Avg Spend</div><div class="fw-bold">$${Number(m.avg_spend || 0).toFixed(2)}</div></div>
                <div class="col-3 text-center"><div class="text-muted small">Avg Txns</div><div class="fw-bold">${Number(m.avg_txns || 0).toFixed(1)}</div></div>
                <div class="col-3 text-center"><div class="text-muted small">Avg Basket</div><div class="fw-bold">$${Number(m.avg_basket || 0).toFixed(2)}</div></div>
            </div>`;
        } else if (type === 'churn') {
            bodyHtml = `<div class="row g-3">
                <div class="col-2 text-center"><div class="text-muted small">Customers</div><div class="fw-bold">${m.customers || 0}</div></div>
                <div class="col-2 text-center"><div class="text-muted small">Avg Spend</div><div class="fw-bold">$${Number(m.avg_spend || 0).toFixed(2)}</div></div>
                <div class="col-2 text-center"><div class="text-muted small">Avg Txns</div><div class="fw-bold">${Number(m.avg_txns || 0).toFixed(1)}</div></div>
                <div class="col-2 text-center"><div class="text-muted small">Avg Basket</div><div class="fw-bold">$${Number(m.avg_basket || 0).toFixed(2)}</div></div>
                <div class="col-2 text-center"><div class="text-muted small">Avg Risk</div><div class="fw-bold">${(Number(m.avg_churn_probability || 0) * 100).toFixed(1)}%</div></div>
                <div class="col-2 text-center"><div class="text-muted small">Risk Range</div><div class="fw-bold">${(Number(m.min_churn_probability || 0) * 100).toFixed(0)}% - ${(Number(m.max_churn_probability || 0) * 100).toFixed(0)}%</div></div>
            </div>`;
        }
        
        // ساخت جدول (مشترک برای هر دو)
        bodyHtml += '<h6 class="mt-3"></h6><div class="table-responsive"><table class="table table-sm align-middle"><thead><tr><th>Household</th><th>Spend</th><th>Txns</th><th>Avg Basket</th><th>R-F-M</th><th>Churn %</th><th>Updated</th></tr></thead><tbody>';
		(j.households_page || []).forEach(h => {
			const profileUrl = "".replace('0', h.household_key);
			bodyHtml += `<tr>
				<td>
					<a href="${profileUrl}" class="d-flex align-items-center text-decoration-none text-dark">
						<i class="fa-solid fa-user text-muted me-2"></i>
						<span>${h.household_key}</span>
					</a>
				</td>
				<td>$${Number(h.total_spend || 0).toFixed(2)}</td>
				<td>${h.total_transactions || 0}</td>
				<td>$${Number(h.avg_basket_value || 0).toFixed(2)}</td>
				<td>${h.recency_score || '-'}-${h.frequency_score || '-'}-${h.monetary_score || '-'}</td>
				<td>${(Number(h.churn_probability || 0) * 100).toFixed(1)}%</td>
				<td>${(h.updated_at || '').toString().slice(0, 10)}</td>
			</tr>`;
		});
        bodyHtml += '</tbody></table></div>';
        
        // ساخت فوتر مودال (مشترک برای هر دو)
        const itemsPerPage = 20;
        const startItem = (p.current_page - 1) * itemsPerPage + 1;
        const endItem = Math.min(p.current_page * itemsPerPage, p.total_items);
        let footerHtml = `<div class="d-flex justify-content-between w-100 align-items-center">
            <span class="text-muted small">Showing ${startItem}-${endItem} of ${p.total_items} records</span>
            <div class="pager" style="margin:0">
                <a href="#" class="page-link ${p.has_previous ? '' : 'disabled'}" data-page="1" id="first-page-btn">&laquo;&laquo; First</a>
                <a href="#" class="page-link ${p.has_previous ? '' : 'disabled'}" data-page="${p.current_page - 1}" id="prev-page-btn">Previous</a>
                <a href="#" class="page-link ${p.has_next ? '' : 'disabled'}" data-page="${p.current_page + 1}" id="next-page-btn">Next</a>
                <a href="#" class="page-link ${p.has_next ? '' : 'disabled'}" data-page="${p.total_pages}" id="last-page-btn">Last &raquo;&raquo;</a>
            </div>
        </div>`;

        // ذخیره اطلاعات لازم برای کلیک‌های بعدی روی خود مودال
        detailModalEl.dataset.type = type;
        detailModalEl.dataset.identifier = identifier;
        
        showDetailModal(titlePrefix + identifier, bodyHtml, footerHtml);
    }
    
    // -----------------------------------------------------------------
    // ۳. Event Listener مرکزی برای مدیریت کلیک دکمه‌های صفحه‌بندی
    // -----------------------------------------------------------------
    detailModalEl.addEventListener('click', function(event) {
        const target = event.target.closest('a'); // برای اطمینان از کلیک روی لینک
        if (!target) return;

        const type = detailModalEl.dataset.type;
        const identifier = detailModalEl.dataset.identifier;

        // Any pager link carrying a page number, so First and Last work like the
        // two that were here before rather than needing their ids listed.
        if (target.dataset.page !== undefined) {
            event.preventDefault();
            if (!target.classList.contains('disabled')) {
                const page = target.dataset.page;
                fetchAndDisplayPage(type, identifier, page);
            }
        }
    });

    // -----------------------------------------------------------------
    // ۴. Event Listenerهای اولیه برای باز کردن مودال‌ها
    // -----------------------------------------------------------------
    document.querySelectorAll('.segment-card').forEach(el => {
        el.addEventListener('click', () => {
            fetchAndDisplayPage('rfm', el.dataset.seg, 1);
        });
    });

    document.querySelectorAll('.churn-card').forEach(el => {
        el.addEventListener('click', () => {
            fetchAndDisplayPage('churn', el.dataset.risk, 1);
        });
    });

    // کد مربوط به مودال Household (که صفحه‌بندی ندارد)
    document.querySelectorAll('.household-card').forEach(el=>{
        el.addEventListener('click', async ()=>{
            const hh = el.dataset.household;
            const form = new FormData(); form.append('csrfmiddlewaretoken', csrftoken); form.append('household_key', hh);
            const res = await fetch('/analysis/api/household/', {method:'POST', body: form});
            const j = await res.json();
            if(j.error){ return showDetailModal('Household '+hh, '<div class="text-danger">'+j.error+'</div>'); }
            const seg = j.segment||{}; 
            let body = `<div class="row g-3">
                <div class="col-2 text-center"><div class="text-muted small">Segment</div><div class="fw-bold">${seg.rfm_segment||'-'}</div></div>
                <div class="col-2 text-center"><div class="text-muted small">R-F-M</div><div class="fw-bold">${seg.recency_score||'-'}-${seg.frequency_score||'-'}-${seg.monetary_score||'-'}</div></div>
                <div class="col-2 text-center"><div class="text-muted small">Churn Risk</div><div class="fw-bold">${(Number(seg.churn_probability||'0')*100).toFixed(1)}%</div></div>
                <div class="col-2 text-center"><div class="text-muted small">Total Spend</div><div class="fw-bold">$${Number(seg.total_spend||0).toFixed(2)}</div></div>
                <div class="col-2 text-center"><div class="text-muted small">Transactions</div><div class="fw-bold">${seg.total_transactions||0}</div></div>
                <div class="col-2 text-center"><div class="text-muted small">Avg Basket</div><div class="fw-bold">$${Number(seg.avg_basket_value||0).toFixed(2)}</div></div>
            </div>`;
            body += '<h6 class="mt-3">Recent Transactions</h6><div class="table-responsive"><table class="table table-sm align-middle"><thead><tr><th>Basket</th><th>Product</th><th>Qty</th><th>Sales</th><th>Day</th></tr></thead><tbody>';
            (j.recent_transactions||[]).forEach(t=>{ body += `<tr><td>${t.basket_id}</td><td>${t.product_id} ${t.commodity_desc?('— '+t.commodity_desc):''}</td><td>${t.quantity}</td><td>$${Number(t.sales_value||0).toFixed(2)}</td><td>${t.day}</td></tr>`; });
            body += '</tbody></table></div>';
            showDetailModal('Household: '+hh, body);
        })
    });
});

// Regenerate Segments Button Handler
const regenerateSegmentsButton = document.getElementById('regenerateSegmentsBtn');
if (regenerateSegmentsButton) regenerateSegmentsButton.addEventListener('click', function() {
    const btn = this;
    const originalHTML = btn.innerHTML;

    // Show loading state
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Regenerating...';

    // Call API to regenerate segments
    fetch('/analysis/api/regenerate-segments/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show success message
            showNotification('Success', `Successfully regenerated ${data.count} customer segments!`, 'success');

            // Reload page after 1.5 seconds to show updated segments
            setTimeout(() => {
                window.location.reload();
            }, 1500);
        } else {
            // Show error message
            showNotification('Error', data.error || 'Failed to regenerate segments', 'error');
            btn.disabled = false;
            btn.innerHTML = originalHTML;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error', 'An error occurred while regenerating segments', 'error');
        btn.disabled = false;
        btn.innerHTML = originalHTML;
    });
});

// Helper function to get CSRF token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const methodSelect = document.getElementById('windowMethod');
const stepControl = document.getElementById('stepControl');
const stepHelp = document.getElementById('stepHelp');
const observationSelect = document.querySelector('[name="observation_window_days"]');
const horizonSelect = document.querySelector('[name="prediction_horizon_days"]');
const stepSelect = document.querySelector('[name="sliding_step_days"]');
const boundaryStatus = document.getElementById('windowBoundaryStatus');
const trainExperimentButton = document.getElementById('runExperimentButton');

function updateBoundaryStatus() {
    const minDay = Number(boundaryStatus.dataset.minDay);
    const maxDay = Number(boundaryStatus.dataset.maxDay);
    const observation = Number(observationSelect.value);
    const horizon = Number(horizonSelect.value);
    const nonOverlapping = methodSelect.value === 'non_overlapping';
    const step = nonOverlapping ? observation + horizon : Number(stepSelect.value);

    if (!Number.isFinite(minDay) || !Number.isFinite(maxDay) || maxDay < minDay || !step) {
        boundaryStatus.className = 'alert alert-warning mt-2 mb-0 py-2 small';
        boundaryStatus.textContent = 'Dataset range is unavailable. Training will validate the selected window settings.';
        trainExperimentButton.disabled = false;
        return;
    }

    const firstEnd = minDay + observation + horizon - 1;
    const completeWindows = firstEnd > maxDay ? 0 : Math.floor((maxDay - firstEnd) / step) + 1;
    const lastEnd = completeWindows ? firstEnd + (completeWindows - 1) * step : minDay - 1;
    const nextEnd = firstEnd + completeWindows * step;
    const missingDays = Math.max(0, nextEnd - maxDay);
    const tailDays = Math.max(0, maxDay - lastEnd);

    if (completeWindows >= 3) {
        boundaryStatus.className = 'alert alert-success mt-2 mb-0 py-2 small';
        boundaryStatus.innerHTML = `<strong>Ready to train.</strong> Dataset days ${minDay}-${maxDay}; ${completeWindows} complete historical windows are available. ${tailDays} trailing day(s) are excluded because their future outcome is incomplete.`;
        trainExperimentButton.disabled = false;
        return;
    }

    trainExperimentButton.disabled = true;
    boundaryStatus.className = 'alert alert-warning mt-2 mb-0 py-2 small';
    let recommendation = 'Choose smaller observation/horizon values.';
    if (nonOverlapping) {
        const suggestedStep = 30;
        const slidingWindows = firstEnd > maxDay ? 0 : Math.floor((maxDay - firstEnd) / suggestedStep) + 1;
        recommendation = `Switch to Sliding Windows with a ${suggestedStep}-day step (${slidingWindows} complete windows). <button type="button" class="btn btn-sm btn-outline-primary ms-2" id="useSlidingRecommendation">Use this setup</button>`;
    }
    boundaryStatus.innerHTML = `<strong>Cannot train this configuration.</strong> Only ${completeWindows} complete window(s) fit in days ${minDay}-${maxDay}. The next window would end on day ${nextEnd}, ${missingDays} day(s) beyond the dataset. ${recommendation}`;
    const recommendationButton = document.getElementById('useSlidingRecommendation');
    if (recommendationButton) {
        recommendationButton.addEventListener('click', () => {
            methodSelect.value = 'sliding';
            stepSelect.value = '30';
            updateWindowControls();
        });
    }
}

function updateWindowControls() {
    const nonOverlapping = methodSelect.value === 'non_overlapping';
    stepControl.classList.toggle('d-none', nonOverlapping);
    stepHelp.textContent = nonOverlapping
        ? 'Step size is calculated automatically: observation window + prediction horizon.'
        : 'Sliding windows overlap. Labels always begin after each cutoff day.';
    updateBoundaryStatus();
}
methodSelect.addEventListener('change', updateWindowControls);
observationSelect.addEventListener('change', updateBoundaryStatus);
horizonSelect.addEventListener('change', updateBoundaryStatus);
stepSelect.addEventListener('change', updateBoundaryStatus);
updateWindowControls();

function refreshDashboard(showNewestExperiment = false) {
    // A cache-busting URL makes sure the new active experiment and its scores
    // are rendered from the server rather than an older browser page.
    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set('refresh', Date.now().toString());
    window.location.assign(nextUrl.toString());
}

document.getElementById('churnExperimentForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    const button = document.getElementById('runExperimentButton');
    const progress = document.getElementById('trainingProgress');
    const startedAt = Date.now();
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training…';
    progress.classList.remove('d-none');
    progress.textContent = 'Checking reusable cache, then training only if needed… 0s elapsed';
    const progressTimer = window.setInterval(() => {
        progress.textContent = `Checking reusable cache, then training only if needed… ${Math.floor((Date.now() - startedAt) / 1000)}s elapsed`;    }, 1000);
    try {
        const response = await fetch('/analysis/api/churn/experiments/run/', {method: 'POST', body: new FormData(this)});
        const payload = await response.json();
        if (!payload.success) throw new Error(payload.error || 'Training failed.');
        const history = payload.history || {};
        const elapsed = Number(payload.experiment.elapsed_seconds || 0).toFixed(1);
        const restored = Boolean(payload.experiment.restored_from_model_cache);
        const message = restored
            ? `Restored the identical saved model result in ${elapsed}s. No model retraining was needed. Recall ${payload.experiment.recall.toFixed(3)}, F1 ${payload.experiment.f1.toFixed(3)}. Activate it when you are ready.`
            : `Trained in ${elapsed}s. Recall ${payload.experiment.recall.toFixed(3)}, F1 ${payload.experiment.f1.toFixed(3)}. Saved ${history.window_records || 0} window-history records, including ${history.scored_records || 0} scored checkpoints. Future identical rules can restore this result without retraining.`;
        showNotification(restored ? 'Experiment restored from cache' : 'Experiment complete', message, 'success');
        window.setTimeout(() => refreshDashboard(true), 1200);
    } catch (error) {
        showNotification('Training failed', error.message, 'error');
        button.disabled = false;
        button.innerHTML = '<i class="fa-solid fa-play me-2"></i>Train experiment';
        progress.classList.add('d-none');
    } finally {
        window.clearInterval(progressTimer);
    }
});

function confirmExperimentAction({title, message, confirmLabel, confirmClass = 'btn-primary'}) {
    const dialog = document.getElementById('experimentConfirmDialog');
    if (!dialog || !dialog.showModal) return Promise.resolve(window.confirm(message));
    return new Promise(resolve => {
        const titleElement = document.getElementById('experimentConfirmTitle');
        const messageElement = document.getElementById('experimentConfirmMessage');
        const cancelButton = document.getElementById('experimentConfirmCancel');
        const acceptButton = document.getElementById('experimentConfirmAccept');
        const finish = accepted => { dialog.close(); resolve(accepted); };
        titleElement.textContent = title;
        messageElement.textContent = message;
        acceptButton.textContent = confirmLabel;
        acceptButton.className = `btn ${confirmClass}`;
        cancelButton.onclick = () => finish(false);
        acceptButton.onclick = () => finish(true);
        dialog.oncancel = event => { event.preventDefault(); finish(false); };
        dialog.showModal();
    });
}

document.querySelectorAll('.activate-experiment').forEach(button => button.addEventListener('click', async function() {
    if (!await confirmExperimentAction({
        title: 'Activate churn rule',
        message: 'Use this experiment to update the dashboard churn-risk scores?',
        confirmLabel: 'Activate rule'
    })) return;
    const response = await fetch(`/analysis/api/churn/experiments/${this.dataset.id}/activate/`, {method: 'POST'});
    const payload = await response.json();
    if (payload.success) {
        showNotification('Experiment activated', 'Dashboard scores were updated. Refreshing the page now...', 'success');
        window.setTimeout(refreshDashboard, 700);
    }
    else showNotification('Activation failed', payload.error || 'Could not activate experiment.', 'error');
}));

function escapeExperimentText(value) {
    return String(value ?? '').replace(/[&<>'"]/g, character => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;'
    })[character]);
}

function formatExperimentMetric(value, digits = 3) {
    return value === null || value === undefined ? 'N/A' : Number(value).toFixed(digits);
}

function showExperimentDetailsModal(body) {
    const modalElement = document.getElementById('detailModal');
    const modalTitle = modalElement?.querySelector('.modal-title');
    const modalBody = modalElement?.querySelector('.modal-body');
    const modalFooter = modalElement?.querySelector('.modal-footer');
    if (!modalElement || !modalTitle || !modalBody || !modalFooter) {
        throw new Error('The details window is not available on this page.');
    }
    modalTitle.textContent = 'Churn experiment details';
    modalBody.innerHTML = body;
    modalFooter.innerHTML = '<button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>';
    bootstrap.Modal.getOrCreateInstance(modalElement).show();
}

document.querySelectorAll('.experiment-details').forEach(button => button.addEventListener('click', async function() {
    try {
        const response = await fetch(`/analysis/api/churn/experiments/${this.dataset.id}/details/`);
        const payload = await response.json();
        if (!payload.success) throw new Error(payload.error || 'Could not load this rule.');
        const experiment = payload.experiment;
        const metadata = experiment.metadata || {};
        const hasMetadata = Object.keys(metadata).length > 0;
        const metadataHtml = hasMetadata
            ? `<div class="experiment-technical"><details><summary><i class="fa-solid fa-sliders me-2"></i>Training definition and technical settings</summary><div class="experiment-technical-content"><p class="small text-muted mb-2">${escapeExperimentText(metadata.label_definition || '')}</p><p class="small text-muted mb-0">${escapeExperimentText(metadata.time_split || '')}</p><div class="experiment-parameter-grid">${Object.entries(metadata.model_parameters || {})
                   .map(([key, value]) => `<div class="experiment-parameter"><strong>${escapeExperimentText(key)}</strong><br>${escapeExperimentText(value)}</div>`)
                   .join('')}</div></div></details></div>`
            : '<div class="experiment-technical"><details><summary><i class="fa-solid fa-sliders me-2"></i>Training definition and technical settings</summary><div class="experiment-technical-content small text-muted">Detailed metadata was not saved for this older rule. Train it again to save full details.</div></details></div>';
        const body = `
            <div class="experiment-detail-rule">
                <div><div class="experiment-detail-eyebrow">Rule configuration</div><h6>${escapeExperimentText(experiment.method)} windows</h6><div class="small text-muted">${experiment.observation_window_days}d observation · ${experiment.prediction_horizon_days}d horizon · ${experiment.step_size_days}d step</div></div>
                <div class="experiment-detail-cutoff"><div class="experiment-detail-eyebrow">Classification cutoff</div><strong>${formatExperimentMetric(experiment.classification_threshold, 2)}</strong><span class="small text-muted">Used for Recall and F1</span></div>
            </div>
            <div class="row g-2">${[['Recall', formatExperimentMetric(experiment.recall)], ['F1', formatExperimentMetric(experiment.f1)], ['PR-AUC', formatExperimentMetric(experiment.pr_auc)], ['ROC-AUC', formatExperimentMetric(experiment.roc_auc)]].map(([label, value]) => `<div class="col-6 col-md-3"><div class="experiment-detail-metric"><div class="label">${label}</div><div class="value">${value}</div></div></div>`).join('')}</div>
            <div class="experiment-detail-section"><div class="experiment-detail-section-title"><i class="fa-solid fa-clipboard-check me-2 text-primary"></i>Test-period check</div><div class="row g-2"><div class="col-6 col-md-3"><div class="experiment-detail-check"><span class="small text-muted">Correct churn (TP)</span><span class="value">${experiment.true_positive ?? '—'}</span></div></div><div class="col-6 col-md-3"><div class="experiment-detail-check"><span class="small text-muted">Missed churners (FN)</span><span class="value">${experiment.false_negative ?? '—'}</span></div></div><div class="col-6 col-md-3"><div class="experiment-detail-check"><span class="small text-muted">False alarms (FP)</span><span class="value">${experiment.false_positive ?? '—'}</span></div></div><div class="col-6 col-md-3"><div class="experiment-detail-check"><span class="small text-muted">Correct non-churn (TN)</span><span class="value">${experiment.true_negative ?? '—'}</span></div></div></div></div>
            <div class="experiment-detail-section"><div class="experiment-detail-section-title"><i class="fa-solid fa-chart-pie me-2 text-primary"></i>Sample split</div><div class="row g-2"><div class="col-4"><div class="experiment-detail-check"><span class="small text-muted">Training</span><span class="value">${experiment.training_samples ?? '—'}</span></div></div><div class="col-4"><div class="experiment-detail-check"><span class="small text-muted">Validation</span><span class="value">${experiment.validation_samples ?? '—'}</span></div></div><div class="col-4"><div class="experiment-detail-check"><span class="small text-muted">Test</span><span class="value">${experiment.test_samples ?? '—'}</span></div></div></div></div>
            ${metadataHtml}`;
        showExperimentDetailsModal(body);
    } catch (error) {
        showNotification('Details unavailable', error.message, 'error');
    }
}));

// Helper function to show notifications
function showNotification(title, message, type) {
    const alertClass = type === 'success' ? 'alert-success' : 'alert-danger';
    const icon = type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle';

    const notification = document.createElement('div');
    notification.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        <i class="fas ${icon} me-2"></i>
        <strong>${title}:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(notification);

    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}
// Add a delete action to every generated-rule row without changing its layout.
const generatedExperiments = JSON.parse(document.getElementById('generatedExperimentData').dataset.experiments);
const experimentResultsTable = document.querySelector('#churnExperimentForm ~ .table-responsive table');
if (experimentResultsTable) {
    experimentResultsTable.querySelector('thead th:last-child').textContent = 'Actions';
}
document.querySelectorAll('#churnExperimentForm ~ .table-responsive tbody tr').forEach((row, index) => {
    const experiment = generatedExperiments[index];
    if (!experiment) return;
    const actionCell = row.lastElementChild;
    const actionGroup = document.createElement('div');
    actionGroup.className = 'experiment-actions';
    while (actionCell.firstChild) actionGroup.appendChild(actionCell.firstChild);
    actionCell.appendChild(actionGroup);
    const deleteButton = document.createElement('button');
    deleteButton.type = 'button';
    deleteButton.className = 'btn btn-sm btn-outline-danger';
    deleteButton.textContent = 'Delete';
    if (experiment.active) {
        deleteButton.disabled = true;
        deleteButton.title = 'Activate another rule before deleting the active rule.';
        actionGroup.appendChild(deleteButton);
        return;
    }
    deleteButton.addEventListener('click', async () => {
        const warning = experiment.active
            ? 'This is the active rule. Activate another rule first, then delete it. Continue?'
            : 'Delete this rule? The reusable RFM/history cache will be kept.';
        if (!await confirmExperimentAction({
            title: 'Delete churn rule',
            message: warning,
            confirmLabel: 'Delete rule',
            confirmClass: 'btn-danger'
        })) return;
        const response = await fetch(`/analysis/api/churn/experiments/${experiment.id}/delete/`, {method: 'POST'});
        const payload = await response.json();
        if (!payload.success) {
            showNotification('Delete unavailable', payload.error || 'Could not delete this rule.', 'error');
            return;
        }
        showNotification('Rule deleted', payload.message, 'success');
        window.setTimeout(refreshDashboard, 600);
    });
    actionGroup.appendChild(deleteButton);
});
