document.addEventListener('DOMContentLoaded', () => {
    const concernInput = document.getElementById('concernInput');
    const qaBtn = document.getElementById('qaBtn');
    const qaLoading = document.getElementById('qaLoading');
    const qaResult = document.getElementById('qaResult');
    const resultPoem = document.getElementById('resultPoem');
    const resultSource = document.getElementById('resultSource');
    const resultAdvice = document.getElementById('resultAdvice');
    const loadingStatus = document.getElementById('loadingStatus');

    qaBtn.addEventListener('click', async () => {
        const concern = concernInput.value.trim();
        
        if (!concern) {
            alert(i18n.qa_no_input);
            return;
        }

        // Reset UI
        qaResult.classList.add('hidden');
        qaLoading.classList.remove('hidden');
        qaBtn.disabled = true;
        loadingStatus.textContent = i18n.qa_status_reading;

        try {
            const response = await fetch('/api/qa', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ concern })
            });

            if (!response.ok) {
                throw new Error('API Error');
            }

            const data = await response.json();
            const OUR_BOOK = "Thi Ca Bình Dân Việt Nam do Nguyễn Tấn Long, Phan Canh sưu tầm và biên soạn, được xuất bản năm 1975";

            // Display result
            resultPoem.textContent = data.poem;
            
            // Render advice as separate divs for each paragraph
            resultAdvice.innerHTML = ''; // Clear previous content
            const paragraphs = data.advice.split('\n').filter(p => p.trim() !== '');
            paragraphs.forEach(text => {
                const pDiv = document.createElement('div');
                pDiv.className = 'advice-paragraph';
                pDiv.textContent = text.trim();
                resultAdvice.appendChild(pDiv);
            });
            
            if (data.source) {
                // Check if it's our book (allow for minor variations in whitespace/case)
                const sourceLower = data.source.toLowerCase();
                const isOurBook = sourceLower.includes("nguyễn tấn long") && 
                                 sourceLower.includes("1975") &&
                                 sourceLower.includes("thi ca bình dân");

                if (isOurBook) {
                    // Do not link or highlight our book
                    resultSource.innerHTML = `${i18n.learn_source_label} <span class="source-plain">${data.source}</span>`;
                } else {
                    // Highlight and link other sources
                    const isUrl = data.source.startsWith('http') || data.source.includes('.com');
                    if (isUrl) {
                        resultSource.innerHTML = `${i18n.learn_source_label} <a href="${data.source}" target="_blank" class="source-highlight">${data.source}</a>`;
                    } else {
                        resultSource.innerHTML = `${i18n.learn_source_label} <span class="source-highlight">${data.source}</span>`;
                    }
                }
            } else {
                resultSource.textContent = '';
            }

            qaLoading.classList.add('hidden');
            qaResult.classList.remove('hidden');
            
            // Scroll to result with a slight delay for entry animation
            setTimeout(() => {
                qaResult.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);

        } catch (error) {
            console.error('QA Error:', error);
            alert(i18n.qa_error);
            qaLoading.classList.add('hidden');
        } finally {
            qaBtn.disabled = false;
        }
    });

    // Handle Enter key (Shift+Enter for newline)
    concernInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            qaBtn.click();
        }
    });
});
