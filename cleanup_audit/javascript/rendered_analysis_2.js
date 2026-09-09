
// Modern Dashboard JavaScript
class DashboardController {
  constructor() {
    this.init();
  }
  
  init() {
    this.animateCounters();
    this.initChart();
    this.addInteractivity();
    this.setupIntersectionObserver();
  }
  
  animateCounters() {
    const counters = document.querySelectorAll('.animate-counter');
    
    counters.forEach(counter => {
      const target = parseInt(counter.dataset.target);
      const increment = target / 100;
      let current = 0;
      
      const updateCounter = () => {
        if (current < target) {
          current += increment;
          if (target >= 1000000) {
            counter.textContent = (current / 1000000).toFixed(2) + 'M';
          } else if (target >= 1000) {
            counter.textContent = (current / 1000).toFixed(1) + 'K';
          } else {
            counter.textContent = Math.floor(current);
          }
          requestAnimationFrame(updateCounter);
        } else {
          if (target >= 1000000) {
            counter.textContent = (target / 1000000).toFixed(2) + 'M';
          } else if (target >= 1000) {
            counter.textContent = (target / 1000).toFixed(1) + 'K';
          } else {
            counter.textContent = target;
          }
        }
      };
      
      setTimeout(() => updateCounter(), 500);
    });
  }
  
  async initChart() {
    const ctx = document.getElementById('trendsChart');
    if (!ctx) return;

    // Fetch real market trends data
    let chartData;
    try {
      const response = await fetch('/analysis/api/market-trends/');
      const data = await response.json();

      // Convert revenue to K dollars (divide by 1000)
      const revenueInK = data.datasets.revenue.map(val => (val / 1000).toFixed(1));

      chartData = {
        labels: data.labels,
        datasets: [
          {
            label: 'Sales Volume ($K)',
            data: revenueInK,
            borderColor: '#667eea',
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            borderWidth: 3,
            tension: 0.4,
            fill: true,
            pointBackgroundColor: '#667eea',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 3,
            pointRadius: 6,
            pointHoverRadius: 8,
            yAxisID: 'y'
          },
          {
            label: 'Unique Customers',
            data: data.datasets.customers,
            borderColor: '#764ba2',
            backgroundColor: 'rgba(118, 75, 162, 0.1)',
            borderWidth: 3,
            tension: 0.4,
            fill: true,
            pointBackgroundColor: '#764ba2',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 3,
            pointRadius: 6,
            pointHoverRadius: 8,
            yAxisID: 'y1'
          }
        ]
      };
    } catch (error) {
      console.error('Failed to load market trends:', error);
      // Fallback to static data if API fails
      chartData = {
        labels: ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6'],
        datasets: [
          {
            label: 'Sales Volume',
            data: [65, 78, 90, 81, 95, 108],
            borderColor: '#667eea',
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            borderWidth: 3,
            tension: 0.4,
            fill: true,
            pointBackgroundColor: '#667eea',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 3,
            pointRadius: 6,
            pointHoverRadius: 8,
            yAxisID: 'y'
          },
          {
            label: 'Customer Acquisition',
            data: [28, 35, 42, 48, 52, 61],
            borderColor: '#764ba2',
            backgroundColor: 'rgba(118, 75, 162, 0.1)',
            borderWidth: 3,
            tension: 0.4,
            fill: true,
            pointBackgroundColor: '#764ba2',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 3,
            pointRadius: 6,
            pointHoverRadius: 8,
            yAxisID: 'y1'
          }
        ]
      };
    }

    new Chart(ctx, {
      type: 'line',
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              usePointStyle: true,
              font: {
                size: 14,
                weight: '600'
              },
              padding: 20
            }
          },
          tooltip: {
            backgroundColor: 'rgba(0,0,0,0.8)',
            titleColor: '#ffffff',
            bodyColor: '#ffffff',
            borderColor: '#667eea',
            borderWidth: 1,
            cornerRadius: 10,
            displayColors: false
          }
        },
        scales: {
          x: {
            grid: {
              display: false
            },
            ticks: {
              font: {
                size: 12,
                weight: '500'
              },
              color: '#718096'
            }
          },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: {
              display: true,
              text: 'Sales Volume ($K)',
              color: '#667eea',
              font: {
                size: 13,
                weight: '600'
              }
            },
            grid: {
              color: 'rgba(0,0,0,0.1)'
            },
            ticks: {
              font: {
                size: 12,
                weight: '500'
              },
              color: '#718096'
            }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
              display: true,
              text: 'Unique Customers',
              color: '#764ba2',
              font: {
                size: 13,
                weight: '600'
              }
            },
            grid: {
              drawOnChartArea: false
            },
            ticks: {
              font: {
                size: 12,
                weight: '500'
              },
              color: '#718096'
            }
          }
        },
        interaction: {
          intersect: false,
          mode: 'index'
        },
        animation: {
          duration: 2000,
          easing: 'easeOutQuart'
        }
      }
    });
  }
  
  addInteractivity() {
    // Add hover effects to metric cards
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
      card.addEventListener('mouseenter', () => {
        card.style.transform = 'translateY(-10px) scale(1.02)';
      });
      
      card.addEventListener('mouseleave', () => {
        card.style.transform = 'translateY(0) scale(1)';
      });
    });
    
    // Add click animations to tool cards
    const toolCards = document.querySelectorAll('.tool-card');
    toolCards.forEach(card => {
      card.addEventListener('click', (e) => {
        e.preventDefault();
        
        // Add click animation
        card.style.transform = 'scale(0.95)';
        setTimeout(() => {
          card.style.transform = '';
          // Navigate after animation
          setTimeout(() => {
            window.location.href = card.href;
          }, 100);
        }, 150);
      });
    });
  }
  
  setupIntersectionObserver() {
    const observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }
      });
    }, observerOptions);
    
    // Observe all fade-in elements
    document.querySelectorAll('.fade-in').forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(30px)';
      el.style.transition = 'all 0.8s ease-out';
      observer.observe(el);
    });
  }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  new DashboardController();
  
  // Add some sparkle effects
  setTimeout(() => {
    const heroSection = document.querySelector('.hero-section');
    if (heroSection) {
      heroSection.classList.add('pulse');
      
      setTimeout(() => {
        heroSection.classList.remove('pulse');
      }, 2000);
    }
  }, 1000);
});
