
    // Modern Navigation Controller
    class NavigationController {
      constructor() {
        this.init();
      }
      
      init() {
        this.setupThemeToggle();
        this.setupMobileNav();
        this.setupActiveLinks();
        this.setupScrollEffects();
      }
      
      setupThemeToggle() {
        const setTheme = (theme) => {
          document.documentElement.setAttribute('data-theme', theme);
          try {
            localStorage.setItem('market.theme', theme);
            localStorage.setItem('theme', theme);
          } catch(e) {}
          
          const icon = document.querySelector('#site-theme-toggle .icon');
          if (icon) {
            icon.className = 'icon fa-solid ' + (theme === 'dark' ? 'fa-sun' : 'fa-moon');
          }
        };
        
        const current = document.documentElement.getAttribute('data-theme') || 'light';
        setTheme(current);
        
        const themeBtn = document.getElementById('site-theme-toggle');
        if (themeBtn) {
          themeBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            setTheme(currentTheme === 'dark' ? 'light' : 'dark');
            
            // Add click animation
            themeBtn.style.transform = 'scale(0.9)';
            setTimeout(() => {
              themeBtn.style.transform = '';
            }, 150);
          });
        }
      }
      
      setupMobileNav() {
        const mobileToggle = document.getElementById('mobileNavToggle');
        const navMiddle = document.getElementById('navMiddle');

        if (mobileToggle && navMiddle) {
          mobileToggle.addEventListener('click', () => {
            navMiddle.classList.toggle('show');
            const icon = mobileToggle.querySelector('i');
            if (icon) {
              icon.classList.toggle('fa-bars');
              icon.classList.toggle('fa-times');
            }
          });

          // Close mobile nav when clicking outside
          document.addEventListener('click', (e) => {
            if (!e.target.closest('.modern-nav')) {
              navMiddle.classList.remove('show');
              const icon = mobileToggle.querySelector('i');
              if (icon) {
                icon.classList.add('fa-bars');
                icon.classList.remove('fa-times');
              }
            }
          });
        }
      }
      
      setupActiveLinks() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
          const href = link.getAttribute('href');
          if (!href) return;
          const linkPath = new URL(href, window.location.href).pathname;
          if (currentPath === linkPath || 
              (linkPath !== '/analysis/' && currentPath.startsWith(linkPath))) {
            link.classList.add('active');
          }
        });
      }
      
      setupScrollEffects() {
        let lastScrollY = window.scrollY;
        const nav = document.querySelector('.modern-nav');
        
        window.addEventListener('scroll', () => {
          const currentScrollY = window.scrollY;
          
          if (currentScrollY > lastScrollY && currentScrollY > 100) {
            // Scrolling down
            nav.style.transform = 'translateY(-100%)';
          } else {
            // Scrolling up
            nav.style.transform = 'translateY(0)';
          }
          
          lastScrollY = currentScrollY;
        });
      }
    }
    
    /* Opening a detail from inside an expanded list replaces what is on screen.
       Anything marked data-modal-restore is remembered first -- its markup, how
       far it was scrolled, and the name of the view it belongs to -- so Back can
       put the reader down on the same card they opened rather than closing the
       list and sending them to the top of the page. */
    const detailModalHistory = [];

    function captureDetailModalState() {
      const modalEl = document.getElementById('detailModal');
      if (!modalEl || !modalEl.classList.contains('show')) return;
      const restorable = modalEl.querySelector('[data-modal-restore]');
      if (!restorable) return;
      detailModalHistory.push({
        title: modalEl.querySelector('.modal-title').innerHTML,
        body: modalEl.querySelector('.modal-body').innerHTML,
        view: restorable.dataset.modalRestore,
        scrollTop: restorable.scrollTop,
      });
    }

    function resetDetailModalHistory() {
      detailModalHistory.length = 0;
      updateDetailModalBack();
    }

    function updateDetailModalBack() {
      const button = document.getElementById('detailModalBack');
      if (button) button.hidden = !detailModalHistory.length;
    }

    function detailModalBack() {
      const state = detailModalHistory.pop();
      const modalEl = document.getElementById('detailModal');
      if (!state || !modalEl) return;
      modalEl.querySelector('.modal-title').innerHTML = state.title;
      modalEl.querySelector('.modal-body').innerHTML = state.body;
      updateDetailModalBack();
      /* The list is put back as markup, so anything bound to its cards with
         addEventListener is gone. The page that owns the view re-binds it, then
         the scroll position is restored -- after re-binding, because that step
         can replace the cards. */
      if (typeof window.onDetailModalRestore === 'function') {
        window.onDetailModalRestore(state.view);
      }
      const restored = modalEl.querySelector('[data-modal-restore]');
      if (restored) restored.scrollTop = state.scrollTop;
    }

    // Shared modal helper
    function showDetailModal(title, bodyHtml) {
      const modalEl = document.getElementById('detailModal');
      if (!modalEl) {
        return alert(title + '\n\n' + bodyHtml.replace(/<[^>]+>/g, ''));
      }

      /* A modal that is not on screen starts a fresh trail. Clearing here as
         well as on hide keeps the Back button honest even if the hide event is
         missed, which the dispose-and-recreate below can cause. */
      if (!modalEl.classList.contains('show')) detailModalHistory.length = 0;
      captureDetailModalState();
      
      // Dispose of any existing modal instance first
      const existingModal = bootstrap.Modal.getInstance(modalEl);
      if (existingModal) {
        existingModal.dispose();
      }
      
      // Clear any leftover backdrop
      const backdrops = document.querySelectorAll('.modal-backdrop');
      backdrops.forEach(backdrop => backdrop.remove());
      
      // Reset body classes and styles
      document.body.classList.remove('modal-open');
      document.body.style.overflow = '';
      document.body.style.paddingRight = '';
      
      modalEl.querySelector('.modal-title').innerHTML = title;
      modalEl.querySelector('.modal-body').innerHTML = bodyHtml;
      updateDetailModalBack();
      
      const modal = new bootstrap.Modal(modalEl, {
        backdrop: true,
        keyboard: true
      });
      
      // Add proper cleanup when modal is hidden
      modalEl.addEventListener('hidden.bs.modal', function() {
        const allBackdrops = document.querySelectorAll('.modal-backdrop');
        allBackdrops.forEach(backdrop => backdrop.remove());
        document.body.classList.remove('modal-open');
        document.body.style.overflow = '';
        document.body.style.paddingRight = '';
        modal.dispose();
      }, { once: true });
      
      modal.show();
      
      // Add fade-in animation to modal content
      const modalBody = modalEl.querySelector('.modal-body');
      modalBody.style.opacity = '0';
      modalBody.style.transform = 'translateY(20px)';
      
      setTimeout(() => {
        modalBody.style.transition = 'all 0.3s ease';
        modalBody.style.opacity = '1';
        modalBody.style.transform = 'translateY(0)';
      }, 100);
    }
    
    // Notification System
    function showNotification(message, type = 'info', duration = 3000) {
      let container = document.getElementById('global-notification-stack');
      if (!container) {
        container = document.createElement('div');
        container.id = 'global-notification-stack';
        Object.assign(container.style, {
          position: 'fixed',
          top: '90px',
          right: '20px',
          zIndex: '9999',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
          maxWidth: '360px'
        });
        document.body.appendChild(container);
      }

      const notification = document.createElement('div');
      notification.className = 'alert alert-' + type + ' alert-dismissible fade show shadow-sm';
      notification.style.margin = '0';
      notification.style.display = 'flex';
      notification.style.alignItems = 'center';
      notification.style.gap = '12px';

      const messageWrapper = document.createElement('div');
      messageWrapper.className = 'flex-grow-1';
      messageWrapper.innerHTML = message;
      notification.appendChild(messageWrapper);

      const closeButton = document.createElement('button');
      closeButton.type = 'button';
      closeButton.className = 'btn-close';
      closeButton.setAttribute('data-bs-dismiss', 'alert');
      closeButton.addEventListener('click', () => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 150);
      });
      notification.appendChild(closeButton);

      container.appendChild(notification);

      setTimeout(() => {
        if (notification) {
          notification.classList.remove('show');
          setTimeout(() => notification.remove(), 200);
        }
      }, duration);
    }
    
    // Initialize when DOM is loaded
    document.addEventListener('DOMContentLoaded', () => {
      new NavigationController();

      document.getElementById('detailModalBack')?.addEventListener('click', detailModalBack);
      /* Closing the modal ends the trail: reopening a detail later should not
         offer to go back to a list the reader has already dismissed. */
      document.getElementById('detailModal')?.addEventListener('hidden.bs.modal',
        resetDetailModalHistory);
      
      // Add smooth scrolling to all anchor links
      document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
          e.preventDefault();
          const target = document.querySelector(this.getAttribute('href'));
          if (target) {
            target.scrollIntoView({
              behavior: 'smooth',
              block: 'start'
            });
          }
        });
      });
    });
  