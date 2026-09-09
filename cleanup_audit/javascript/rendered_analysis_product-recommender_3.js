
document.querySelectorAll('.level-select').forEach(function (select) {
    select.addEventListener('change', function () {
        if (!this.value) {
            return;
        }
        var queryInput = document.getElementById('id_query');
        if (queryInput) {
            queryInput.value = this.value;
            queryInput.focus();
        }
    });
});
