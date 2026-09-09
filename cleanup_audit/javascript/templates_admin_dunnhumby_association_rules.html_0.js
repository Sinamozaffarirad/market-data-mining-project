
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('rulesForm');
    const generateBtn = document.getElementById('generateBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    if (form && generateBtn) {
        form.addEventListener('submit', function(e) {
            // Show loading state
            generateBtn.style.display = 'none';
            loadingIndicator.style.display = 'flex';
            
            // Add timeout safety to prevent infinite loading
            setTimeout(function() {
                generateBtn.style.display = 'inline-flex';
                loadingIndicator.style.display = 'none';
                console.log('Association rules generation timeout - UI reset');
            }, 120000); // 2 minutes timeout
            
            console.log('Association rules generation started');
        });
    }
    
    // Enhanced table interactions
    const rows = document.querySelectorAll('.rule-row');
    rows.forEach(row => {
        row.addEventListener('click', function() {
            // Add click functionality if needed
            this.style.backgroundColor = 'var(--bg-hover)';
            setTimeout(() => {
                this.style.backgroundColor = '';
            }, 200);
        });
    });
});
