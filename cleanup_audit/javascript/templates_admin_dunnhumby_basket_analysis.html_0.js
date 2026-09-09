
document.addEventListener('DOMContentLoaded', function() {
    // Create basket value distribution chart
    const ctx = document.getElementById('basketValueChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['$0-25', '$25-50', '$50-100', '$100-200', '$200-500', '$500+'],
            datasets: [{
                label: 'Number of Baskets',
                data: [245, 189, 167, 134, 89, 45],
                backgroundColor: [
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(139, 92, 246, 0.8)',
                    'rgba(236, 72, 153, 0.8)'
                ],
                borderColor: [
                    'rgba(59, 130, 246, 1)',
                    'rgba(16, 185, 129, 1)',
                    'rgba(245, 158, 11, 1)',
                    'rgba(239, 68, 68, 1)',
                    'rgba(139, 92, 246, 1)',
                    'rgba(236, 72, 153, 1)'
                ],
                borderWidth: 2,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
});

function refreshAnalysis() {
    // Add loading state
    const btn = event.target;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
    btn.disabled = true;
    
    // Simulate API call
    setTimeout(() => {
        location.reload();
    }, 2000);
}

function exportData() {
    alert('Data export functionality would be implemented here');
}

function generateReport() {
    alert('Report generation functionality would be implemented here');
}
