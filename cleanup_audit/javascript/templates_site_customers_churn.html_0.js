
document.addEventListener('click', async function (event) {
    const button = event.target.closest('.build-history-cache');
    if (!button) return;
    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = 'Building history…';
    try {
        const response = await fetch(`/analysis/api/churn/experiments/${button.dataset.experimentId}/cache-history/`, {method: 'POST'});
        const payload = await response.json();
        if (!payload.success) throw new Error(payload.error || 'Could not build the history cache.');
        window.location.reload();
    } catch (error) {
        button.disabled = false;
        button.textContent = originalText;
        window.alert(error.message);
    }
});
