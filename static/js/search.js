document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const resultsDiv = document.getElementById('results');

    function performSearch() {
        const query = searchInput.value.trim();

        if (query.length === 0) {
            resultsDiv.innerHTML = `<div class="no-results" aria-live="polite">${i18n.search_no_input}</div>`;
            return;
        }

        // 20s Countdown & Progress Bar logic
        resultsDiv.innerHTML = `
            <div class="search-progress">
                <div class="progress-info">
                    <span class="status-text">${i18n.search_status_finding}</span>
                    <span class="countdown-timer">20s</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" id="searchProgressFill"></div>
                </div>
            </div>
        `;

        const progressFill = document.getElementById('searchProgressFill');
        const timerText = document.querySelector('.countdown-timer');
        let progress = 0;
        let timeLeft = 20;

        // Slow progress: 0 to 90% over 20 seconds
        const intervalTime = 100; // update every 0.1s
        const totalSteps = (20 * 1000) / intervalTime;
        const progressIncrement = 90 / totalSteps;

        const progressInterval = setInterval(() => {
            progress += progressIncrement;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + '%';
        }, intervalTime);

        const timerInterval = setInterval(() => {
            timeLeft--;
            if (timeLeft < 0) timeLeft = 0;
            timerText.textContent = timeLeft + 's';
        }, 1000);

        fetch('/search?q=' + encodeURIComponent(query))
            .then(response => response.json())
            .then(data => {
                // Clear slow intervals
                clearInterval(progressInterval);
                clearInterval(timerInterval);

                // Speed up to 100%
                progressFill.style.transition = 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
                progressFill.style.width = '100%';
                timerText.textContent = '0s';

                // Brief pause for the animation to finish
                setTimeout(() => {
                    if (data.results && data.results.length > 0) {
                        resultsDiv.innerHTML = data.results.map(item => `
                            <div class="result-item">
                                <div class="poem">${escapeHtml(item.poem)}</div>
                                <div class="meaning">${escapeHtml(item.meaning)}</div>
                                ${item.keywords ? `<div class="keywords">${i18n.keyword_label}${escapeHtml(item.keywords)}</div>` : ''}
                                <div class="match-info">
                                    ${item.category ? `<span class="category-badge">${escapeHtml(item.category)}</span>` : ''}
                                    <span class="keyword-badge">${i18n.matched_label}"${escapeHtml(item.matched_keyword)}"</span>
                                    <span class="accuracy">${i18n.accuracy_label}${(item.score * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        `).join('');
                    } else {
                        resultsDiv.innerHTML = `<div class="no-results" aria-live="polite">${i18n.search_no_results}</div>`;
                    }
                }, 600);
            })
            .catch(err => {
                clearInterval(progressInterval);
                clearInterval(timerInterval);
                resultsDiv.innerHTML = `<div class="no-results" aria-live="polite">${i18n.search_error}</div>`;
                console.error('Search error:', err);
            });
    }

    if (searchBtn) searchBtn.addEventListener('click', performSearch);
    if (searchInput) searchInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') performSearch();
    });

    // Handle suggested search clicks
    const exampleBtns = document.querySelectorAll('.example-btn');
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', function () {
            searchInput.value = this.textContent.trim();
            searchInput.focus();
        });
    });

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
