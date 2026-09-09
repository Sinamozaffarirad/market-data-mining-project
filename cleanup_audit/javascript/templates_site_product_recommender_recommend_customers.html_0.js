
    document.querySelectorAll('.recommendation-action').forEach((link) => {
        link.addEventListener('click', () => {
            document.getElementById('recommenderLoadingOverlay').classList.add('is-visible');
        });
    });
