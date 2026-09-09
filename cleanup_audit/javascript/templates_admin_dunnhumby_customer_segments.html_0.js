
function toggleView(viewType, buttonElement) {
    console.log('Toggling view to:', viewType);
    
    const tableView = document.getElementById('tableView');
    const cardView = document.getElementById('cardView');
    const buttons = document.querySelectorAll('.view-controls .btn');
    
    if (!tableView || !cardView) {
        console.error('View elements not found');
        return;
    }
    
    // Reset button states
    buttons.forEach(btn => {
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-secondary');
    });
    
    if (viewType === 'table') {
        tableView.style.display = 'block';
        cardView.style.display = 'none';
        if (buttonElement) {
            buttonElement.classList.remove('btn-secondary');
            buttonElement.classList.add('btn-primary');
        }
    } else if (viewType === 'cards') {
        tableView.style.display = 'none';
        cardView.style.display = 'grid';
        if (buttonElement) {
            buttonElement.classList.remove('btn-secondary');
            buttonElement.classList.add('btn-primary');
        }
    }
    
    console.log('View toggled successfully');
}

// Enhanced table interactions
document.addEventListener('DOMContentLoaded', function() {
    const rows = document.querySelectorAll('.segment-row, .customer-row');
    rows.forEach(row => {
        row.addEventListener('click', function() {
            // Add click functionality if needed
            this.style.backgroundColor = 'var(--bg-hover)';
            setTimeout(() => {
                this.style.backgroundColor = '';
            }, 200);
        });
    });
    
    // Default to table view
    const tableBtn = document.querySelector('.view-controls .btn');
    if (tableBtn) {
        tableBtn.classList.remove('btn-secondary');
        tableBtn.classList.add('btn-primary');
    }
});
