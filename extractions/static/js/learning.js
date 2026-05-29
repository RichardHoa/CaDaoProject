document.addEventListener('DOMContentLoaded', function () {
    let learningData = [];
    let currentTopicIndex = 0;
    let currentPoemIndex = 0;

    const themeSelect = document.getElementById('themeSelect');
    const learningContainer = document.getElementById('learningContainer');

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    if (themeSelect && learningContainer) {
        fetch('/api/learning/data')
            .then(response => response.json())
            .then(data => {
                learningData = data;
                initLearning();
            })
            .catch(err => {
                learningContainer.innerHTML = `<div class="no-results" aria-live="polite">${i18n.learn_error_data}</div>`;
                console.error(err);
            });
    }

    function getGlobalPoemId() {
        let globalId = 0;
        for (let i = 0; i < currentTopicIndex; i++) {
            globalId += learningData[i].poems.length;
        }
        globalId += currentPoemIndex;
        return globalId;
    }

    function initLearning() {
        themeSelect.innerHTML = learningData.map((topic, i) =>
            `<option value="${i}">${escapeHtml(topic.topic)}</option>`
        ).join('');

        themeSelect.addEventListener('change', (e) => {
            currentTopicIndex = parseInt(e.target.value);
            currentPoemIndex = 0;
            renderPoem();
        });

        renderPoem();
    }

    function renderPoem() {
        if (!learningData || learningData.length === 0) return;

        const topicData = learningData[currentTopicIndex];
        const poemData = topicData.poems[currentPoemIndex];

        let poemHtml = escapeHtml(poemData.poem_text);

        poemData.blanks.forEach((blank, index) => {
            const slotHtml = `<span class="blank-slot empty" data-answer="${escapeHtml(blank.word)}" data-index="${index}" tabindex="0" role="button" aria-label="${i18n.learn_fill_aria}"></span>`;
            poemHtml = poemHtml.replace(escapeHtml(blank.word), slotHtml);
        });

        learningContainer.innerHTML = `
            <div class="learning-intro" id="learningIntro">
                <h2>${i18n.learn_intro_title}</h3>
                <p>${escapeHtml(poemData.introduction || '')}</p>
            </div>
            <div class="learning-poem">${poemHtml.replace(/\n/g, '<br>')}</div>
            <div class="word-bank" id="wordBankContainer" aria-live="polite"></div>
            <div class="learning-actions">
                <button class="check-ans-btn" id="checkAnsBtn">${i18n.learn_check_btn}</button>
                <button class="next-poem-btn" id="nextPoemBtn">${i18n.learn_next_btn}</button>
            </div>
            <div class="learning-feedback" id="learningFeedback" aria-live="polite">
                <div class="author-interpretation">
                    <h2>${i18n.learn_interp_title}</h3>
                    <p>${escapeHtml(poemData.interpretation || '')}</p>
                </div>
                
                <div class="feedback-vote-actions" id="feedbackVoteActions">
                    <p>${i18n.learn_feedback_prompt}</p>
                    <div class="vote-buttons">
                        <button class="vote-btn like-btn" id="likeBtn" aria-label="${i18n.learn_like_aria}">
                            👍 <span>${i18n.learn_like}</span>
                        </button>
                        <button class="vote-btn dislike-btn" id="dislikeBtn" aria-label="${i18n.learn_dislike_aria}">
                            👎 <span>${i18n.learn_dislike}</span>
                        </button>
                    </div>
                </div>

                <div id="interpretationHistory" class="interpretation-history" style="display: none;">
                    <hr>
                    <h3>${i18n.learn_history_title}</h3>
                    <div id="historyList" class="history-list"></div>
                    
                    <div class="user-interpretation-form" id="userInterpForm">
                        <p><strong>${i18n.learn_own_prompt}</strong></p>
                        <div class="form-group">
                            <label for="userNameInput">${i18n.learn_name_label}</label>
                            <input type="text" id="userNameInput" placeholder="${i18n.learn_name_placeholder}" aria-label="${i18n.learn_name_aria}">
                        </div>
                        <div class="form-group">
                            <label for="userInterpInput">${i18n.learn_interp_label}</label>
                            <textarea id="userInterpInput" rows="3" placeholder="${i18n.learn_interp_placeholder}" aria-label="${i18n.learn_interp_aria}"></textarea>
                        </div>
                        <button class="submit-interp-btn" id="submitInterpBtn">${i18n.learn_submit_btn}</button>
                    </div>
                </div>
            </div>
        `;

        document.getElementById('checkAnsBtn').addEventListener('click', checkAnswers);
        document.getElementById('nextPoemBtn').addEventListener('click', nextPoem);
        document.getElementById('likeBtn').addEventListener('click', () => handleFeedback('like'));
        document.getElementById('dislikeBtn').addEventListener('click', () => handleFeedback('dislike'));
        document.getElementById('submitInterpBtn').addEventListener('click', submitInterpretation);

        const slots = document.querySelectorAll('.blank-slot');
        slots.forEach(slot => {
            slot.addEventListener('click', () => handleSlotClick(slot));
            slot.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleSlotClick(slot);
                }
            });
        });

        updateWordBankAndSlots();
    }

    function handleSlotClick(slot) {
        if (slot.classList.contains('empty')) return;
        if (slot.classList.contains('correct')) return;

        slot.textContent = '';
        slot.classList.add('empty');
        slot.classList.remove('incorrect');
        
        updateWordBankAndSlots();
    }

    function updateWordBankAndSlots() {
        const slots = Array.from(document.querySelectorAll('.blank-slot'));
        const wordBankContainer = document.getElementById('wordBankContainer');
        
        const activeSlot = slots.find(slot => slot.classList.contains('empty'));
        
        slots.forEach(slot => {
            slot.classList.remove('active');
            slot.setAttribute('aria-pressed', 'false');
        });

        if (activeSlot) {
            activeSlot.classList.add('active');
            activeSlot.setAttribute('aria-pressed', 'true');
            activeSlot.focus();
            
            const index = parseInt(activeSlot.dataset.index);
            const poemData = learningData[currentTopicIndex].poems[currentPoemIndex];
            const blankData = poemData.blanks[index];
            
            let options = [...new Set(blankData.options)].sort(() => Math.random() - 0.5);
            
            wordBankContainer.innerHTML = options.map((opt, i) =>
                `<button class="word-btn" data-word="${escapeHtml(opt)}" aria-label="${i18n.select_word_aria} ${escapeHtml(opt)}">${escapeHtml(opt)}</button>`
            ).join('');
            
            const wordBtns = document.querySelectorAll('.word-btn');
            wordBtns.forEach(btn => {
                btn.onclick = () => {
                    activeSlot.textContent = btn.dataset.word;
                    activeSlot.classList.remove('empty', 'incorrect', 'active');
                    updateWordBankAndSlots();
                };
            });
        } else {
            wordBankContainer.innerHTML = `<div style="color: #4CAF50; font-weight:bold; width:100%; text-align:center;">${i18n.learn_all_filled}</div>`;
        }
    }

    function checkAnswers() {
        const slots = document.querySelectorAll('.blank-slot');
        let allCorrect = true;

        slots.forEach(slot => {
            const correctAnswer = slot.dataset.answer;
            const userAnswer = slot.textContent;

            if (slot.classList.contains('empty')) {
                allCorrect = false;
                return;
            }

            if (userAnswer === correctAnswer) {
                slot.classList.remove('incorrect');
                slot.classList.add('correct');
                slot.setAttribute('aria-invalid', 'false');
            } else {
                slot.classList.remove('correct');
                slot.classList.add('incorrect');
                slot.setAttribute('aria-invalid', 'true');
                allCorrect = false;
            }
        });

        if (allCorrect) {
            document.getElementById('checkAnsBtn').style.display = 'none';
            document.getElementById('wordBankContainer').style.display = 'none';
            document.getElementById('nextPoemBtn').style.display = 'block';
            document.getElementById('learningFeedback').style.display = 'block';

            // Add citation asterisk at the end of the poem
            const poemDiv = document.querySelector('.learning-poem');
            if (poemDiv && !document.querySelector('.citation-mark')) {
                const asterisk = document.createElement('span');
                asterisk.className = 'citation-mark';
                asterisk.textContent = ' *';
                asterisk.setAttribute('role', 'button');
                asterisk.setAttribute('tabindex', '0');
                asterisk.setAttribute('aria-label', i18n.learn_source_label);
                
                const sourceDiv = document.createElement('div');
                sourceDiv.className = 'source-info';
                sourceDiv.style.display = 'none';
                sourceDiv.innerHTML = `<strong>${i18n.learn_source_label}</strong> ${i18n.disclaimer_text2} ${i18n.disclaimer_text3}`;
                
                asterisk.onclick = () => {
                    const isHidden = sourceDiv.style.display === 'none';
                    sourceDiv.style.display = isHidden ? 'block' : 'none';
                    if (isHidden) sourceDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                };

                asterisk.onkeypress = (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        asterisk.onclick();
                    }
                };
                
                poemDiv.appendChild(asterisk);
                poemDiv.after(sourceDiv);
            }
        } else {
            // Show error message in word bank area
            const wordBankContainer = document.getElementById('wordBankContainer');
            if (wordBankContainer) {
                wordBankContainer.innerHTML = `<div style="color: var(--color-wine); font-weight:bold; width:100%; text-align:center; animation: shake 0.5s;">${i18n.learn_wrong_word}</div>`;
            }
        }
    }

    function handleFeedback(status) {
        const poemId = getGlobalPoemId();
        const voteActions = document.getElementById('feedbackVoteActions');
        
        if (status === 'like') {
            // Simply record the like
            fetch('/api/learning/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ poem_id: poemId, status: 'like' })
            });
            voteActions.innerHTML = `<p style="color: var(--color-green); font-weight: bold;">${i18n.learn_thanks_like}</p>`;
        } else {
            // Record dislike and show other interpretations
            fetch('/api/learning/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ poem_id: poemId, status: 'dislike' })
            });
            voteActions.style.display = 'none';
            showInterpretationHistory();
        }
    }

    function showInterpretationHistory() {
        const poemId = getGlobalPoemId();
        const historyContainer = document.getElementById('interpretationHistory');
        const historyList = document.getElementById('historyList');
        
        historyContainer.style.display = 'block';
        historyList.innerHTML = `<div class="loading">${i18n.learn_loading_history}</div>`;

        fetch(`/api/learning/interpretations/${poemId}`)
            .then(res => res.json())
            .then(data => {
                if (data.results && data.results.length > 0) {
                    historyList.innerHTML = data.results.map(item => `
                        <div class="history-item">
                            <div class="item-header">
                                <span class="username">${escapeHtml(item.username)}</span>
                                <span class="status-tag ${item.status}">${item.status === 'like' ? i18n.learn_like : i18n.learn_dislike}</span>
                            </div>
                            <p class="content">${escapeHtml(item.interpretation || '')}</p>
                        </div>
                    `).join('');
                } else {
                    historyList.innerHTML = `<p class="no-history">${i18n.learn_no_history}</p>`;
                }
                // Scroll to history
                historyContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            })
            .catch(err => {
                console.error(err);
                historyList.innerHTML = `<p class="error">${i18n.learn_error_history}</p>`;
            });
    }

    function submitInterpretation() {
        const poemId = getGlobalPoemId();
        const username = document.getElementById('userNameInput').value.trim();
        const interpretation = document.getElementById('userInterpInput').value.trim();
        const submitBtn = document.getElementById('submitInterpBtn');

        if (!interpretation) {
            alert(i18n.learn_empty_interp);
            return;
        }

        submitBtn.disabled = true;
        submitBtn.textContent = i18n.learn_submitting;

        fetch('/api/learning/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                poem_id: poemId,
                status: 'dislike',
                username: username,
                interpretation: interpretation
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                document.getElementById('userInterpForm').innerHTML = `<p style="color: var(--color-green); font-weight: bold; padding: 1rem; background: #f0fdf4; border-radius: 0.5rem;">${i18n.learn_thanks_submit}</p>`;
                showInterpretationHistory(); // Refresh list
            }
        })
        .catch(err => {
            console.error(err);
            alert(i18n.learn_error_submit);
            submitBtn.disabled = false;
            submitBtn.textContent = i18n.learn_submit_btn;
        });
    }

    function nextPoem() {
        const topicData = learningData[currentTopicIndex];
        currentPoemIndex++;

        if (currentPoemIndex >= topicData.poems.length) {
            learningContainer.innerHTML = `
                <div class="no-results">
                    <h3>${i18n.learn_congrats_title}</h3>
                    <p>${i18n.learn_congrats_desc}</p>
                </div>
            `;
        } else {
            renderPoem();
        }
    }
});
