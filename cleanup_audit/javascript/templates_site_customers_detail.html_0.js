
  document.querySelectorAll("a[href*='recommendations']").forEach((recBtn) => {
    recBtn.addEventListener("click", function(){
      document.getElementById('recommenderLoadingOverlay').classList.add('is-visible');
    });
  });
