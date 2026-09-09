
    (function(){
      try{
        var t = localStorage.getItem('market.theme') || 'light';
        document.documentElement.setAttribute('data-theme', t);
        localStorage.setItem('theme', t); // keep django key aligned even outside admin
      }catch(e){}
    })();
  