document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const categorySelect = document.getElementById('categorySelect');
    const resultsDiv = document.getElementById('results');

    function performSearch() {
        const query = searchInput.value.trim();
        const category = categorySelect ? categorySelect.value : 'All';

        if (query.length === 0) {
            resultsDiv.innerHTML = `<div class="no-results" aria-live="polite">${i18n.search_no_input}</div>`;
            return;
        }

        // 3s Progress Bar logic
        resultsDiv.innerHTML = `
            <div class="search-progress">
                <div class="progress-info">
                    <span class="status-text">${i18n.search_status_finding}</span>
                    <span class="countdown-timer">3s</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" id="searchProgressFill"></div>
                </div>
            </div>
        `;

        const progressFill = document.getElementById('searchProgressFill');
        const timerText = document.querySelector('.countdown-timer');
        let progress = 0;
        let timeLeft = 3;

        // Animate from 0 to 90% over 3 seconds (30 steps of 100ms)
        const intervalTime = 100;
        const totalSteps = 3000 / intervalTime;
        const progressIncrement = 90 / totalSteps;

        const progressInterval = setInterval(() => {
            progress += progressIncrement;
            if (progress > 90) progress = 90;
            if (progressFill) progressFill.style.width = progress + '%';
        }, intervalTime);

        const timerInterval = setInterval(() => {
            timeLeft--;
            if (timeLeft < 0) timeLeft = 0;
            if (timerText) timerText.textContent = timeLeft + 's';
        }, 1000);

        fetch('/search?q=' + encodeURIComponent(query) + '&cat=' + encodeURIComponent(category))
            .then(response => response.json())
            .then(data => {
                // Clear intervals
                clearInterval(progressInterval);
                clearInterval(timerInterval);

                // Instantly animate to 100%
                if (progressFill) {
                    progressFill.style.transition = 'width 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
                    progressFill.style.width = '100%';
                }
                if (timerText) timerText.textContent = '0s';

                // Brief pause for the transition animation
                setTimeout(() => {
                    if (data.results && data.results.length > 0) {
                        resultsDiv.innerHTML = data.results.map(item => {
                            const formattedPoem = escapeHtml(item.poem).replace(/\n/g, '<br>');
                            const formattedMeaning = escapeHtml(item.meaning).replace(/\n/g, '<br>');
                            const badgeClass = cleanCategoryForClass(item.category);
                            
                            return `
                                <div class="result-card">
                                    <div class="result-card-left">
                                        <span class="category-badge badge-${badgeClass}">${escapeHtml(item.category)}</span>
                                        <div class="poem">${formattedPoem}</div>
                                        <div class="meaning">${formattedMeaning}</div>
                                        ${item.keywords ? `<div class="keywords">${i18n.keyword_label}${escapeHtml(item.keywords)}</div>` : ''}
                                        <div class="match-info">
                                            <span class="keyword-badge">${i18n.matched_label}"${escapeHtml(item.matched_keyword)}"</span>
                                            <span class="accuracy">${i18n.accuracy_label}${(item.score * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                    ${item.id ? `
                                    <div class="result-card-right">
                                        <div class="image-wrapper">
                                            <div class="image-skeleton"></div>
                                            <img 
                                                src="/wiki/image/${item.id}" 
                                                alt="${escapeHtml(item.category)}" 
                                                loading="lazy"
                                                onload="this.classList.add('loaded'); this.previousElementSibling.style.display='none';"
                                                onerror="this.style.display='none'; this.previousElementSibling.style.display='none';"
                                            >
                                        </div>
                                    </div>` : ''}
                                </div>
                            `;
                        }).join('');
                    } else {
                        resultsDiv.innerHTML = `<div class="no-results" aria-live="polite">${i18n.search_no_results}</div>`;
                    }
                }, 400);
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
    if (categorySelect) categorySelect.addEventListener('change', performSearch);

    // Handle suggested search clicks
    const exampleBtns = document.querySelectorAll('.example-btn');
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', function () {
            searchInput.value = this.textContent.trim();
            searchInput.focus();
            performSearch();
        });
    });

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function cleanCategoryForClass(category) {
        if (!category) return 'default';
        return category.replace(/[()]/g, '').trim().toLowerCase().replace(/\s+/g, '-');
    }

    // --- Lightbox Overlay ---
    const lightbox = document.getElementById('wikiLightbox');
    const lightboxImg = document.getElementById('lightboxImg');
    const lightboxClose = document.getElementById('lightboxClose');

    function openLightbox(src) {
        if (lightboxImg && lightbox) {
            lightboxImg.src = src;
            lightbox.classList.remove("hidden");
            document.body.style.overflow = "hidden";
        }
    }

    function closeLightbox() {
        if (lightbox && lightboxImg) {
            lightbox.classList.add("hidden");
            lightboxImg.src = "";
            document.body.style.overflow = "";
        }
    }

    if (resultsDiv) {
        resultsDiv.addEventListener("click", function(e) {
            const img = e.target.closest("img");
            if (img && img.closest(".image-wrapper")) {
                openLightbox(img.src);
            }
        });
    }

    if (lightboxClose) lightboxClose.addEventListener("click", closeLightbox);
    if (lightbox) {
        lightbox.addEventListener("click", function(e) {
            if (e.target === lightbox) {
                closeLightbox();
            }
        });
    }

    document.addEventListener("keydown", function(e) {
        if (e.key === "Escape" && lightbox && !lightbox.classList.contains("hidden")) {
            closeLightbox();
        }
    });
});
