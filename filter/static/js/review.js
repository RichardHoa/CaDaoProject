// review.js
document.addEventListener('DOMContentLoaded', () => {
    let currentId = 1;
    let totalCount = 8704;
    let isDeleted = false;
    
    // State arrays
    let masterKeywords = [];
    let selectedKeywords = [];
    let originalData = { poem: '', category: '', explanation: '', book: '', keyword: '', type: '' };

    // Elements
    const progressBar = document.getElementById('progress-bar');
    const reviewedCountSpan = document.getElementById('reviewed-count');
    const totalCountSpan = document.getElementById('total-count');
    const progressPercentSpan = document.getElementById('progress-percent');
    
    const jumpInput = document.getElementById('jump-input');
    const jumpBtn = document.getElementById('jump-btn');
    const addBtn = document.getElementById('add-btn');

    const poemIdSpan = document.getElementById('poem-id');
    const categorySelect = document.getElementById('poem-category');
    const bookSelect = document.getElementById('poem-book');
    const typeSelect = document.getElementById('poem-type');
    const addTypeBtn = document.getElementById('add-type-btn');
    
    // Keyword chips elements
    const chipsContainer = document.getElementById('chips-container');
    const keywordSearchInput = document.getElementById('keyword-search-input');
    const keywordDropdown = document.getElementById('keyword-dropdown');
    const addKeywordBtn = document.getElementById('add-keyword-btn');
    
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = statusIndicator.querySelector('.status-text');
    
    const poemTextarea = document.getElementById('poem-text');
    const explanationTextarea = document.getElementById('poem-explanation');
    const syllableDisplay = document.getElementById('syllable-display');

    const prevBtn = document.getElementById('prev-btn');
    const deleteBtn = document.getElementById('delete-btn');
    const saveBtn = document.getElementById('save-btn');
    const nextBtn = document.getElementById('next-btn');
    
    const prevIdText = document.getElementById('prev-id-text');
    const nextIdText = document.getElementById('next-id-text');

    // Startup flow: Load types & keywords, then load poem
    initApp();

    async function initApp() {
        await loadTypes();
        await loadKeywords();
        const state = await fetchProgress();
        currentId = state.current_id || 1;
        loadPoem(currentId);
    }

    // Event Listeners
    prevBtn.addEventListener('click', () => navigateTo(currentId - 1));
    nextBtn.addEventListener('click', () => navigateTo(currentId + 1));
    saveBtn.addEventListener('click', saveCurrentPoem);
    deleteBtn.addEventListener('click', deleteCurrentPoem);
    addBtn.addEventListener('click', addNewPoem);
    
    jumpBtn.addEventListener('click', () => {
        const val = parseInt(jumpInput.value);
        if (val && val >= 1) {
            navigateTo(val);
        } else {
            alert("Vui lòng nhập ID hợp lệ!");
        }
    });

    jumpInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            jumpBtn.click();
        }
    });

    addTypeBtn.addEventListener('click', async () => {
        const newType = prompt('Nhập thể loại phụ mới (Ví dụ: Khuyên bảo, Châm biếm, Tình cảm...):');
        if (newType && newType.trim()) {
            const cleanType = newType.trim();
            try {
                const res = await fetch('/api/types', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ type: cleanType })
                });
                
                if (res.ok) {
                    const types = await res.json();
                    populateTypesDropdown(types);
                    typeSelect.value = cleanType;
                    checkModified();
                } else {
                    const err = await res.json();
                    alert(err.error || 'Không thể thêm thể loại phụ!');
                }
            } catch (err) {
                console.error(err);
                alert('Lỗi kết nối khi thêm thể loại phụ.');
            }
        }
    });

    // Keyword Selector Event Listeners
    keywordSearchInput.addEventListener('focus', filterKeywordDropdown);
    keywordSearchInput.addEventListener('input', filterKeywordDropdown);
    
    // Hide dropdown when input is blurred, after short delay to let click register
    keywordSearchInput.addEventListener('blur', () => {
        setTimeout(() => {
            keywordDropdown.style.display = 'none';
        }, 150);
    });

    addKeywordBtn.addEventListener('click', async () => {
        const newKw = prompt('Nhập từ khóa mới vào hệ thống (Ví dụ: cha mẹ, tình yêu, châm biếm...):');
        if (newKw && newKw.trim()) {
            const cleanKw = newKw.trim().toLowerCase();
            try {
                const res = await fetch('/api/keywords', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ keyword: cleanKw })
                });
                
                if (res.ok) {
                    masterKeywords = await res.json();
                    // Automatically select it as a chip if not already selected
                    if (!selectedKeywords.includes(cleanKw)) {
                        selectedKeywords.push(cleanKw);
                        renderChips();
                        checkModified();
                    }
                    keywordSearchInput.value = '';
                } else {
                    const err = await res.json();
                    alert(err.error || 'Không thể thêm từ khóa mới!');
                }
            } catch (err) {
                console.error(err);
                alert('Lỗi kết nối khi thêm từ khóa.');
            }
        }
    });

    // Detect changes to update modified status
    [categorySelect, bookSelect, typeSelect, poemTextarea, explanationTextarea].forEach(el => {
        el.addEventListener('input', checkModified);
        el.addEventListener('change', checkModified);
    });

    // Real-time poem syllable counting
    poemTextarea.addEventListener('input', updateSyllables);

    // Functions
    async function loadTypes() {
        try {
            const res = await fetch('/api/types');
            const types = await res.json();
            populateTypesDropdown(types);
        } catch (err) {
            console.error('Error loading types list:', err);
        }
    }

    function populateTypesDropdown(types) {
        typeSelect.innerHTML = '';
        types.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t;
            typeSelect.appendChild(opt);
        });
    }

    async function loadKeywords() {
        try {
            const res = await fetch('/api/keywords');
            masterKeywords = await res.json();
        } catch (err) {
            console.error('Error loading keywords list:', err);
        }
    }

    function renderChips() {
        chipsContainer.innerHTML = '';
        selectedKeywords.forEach(kw => {
            const chip = document.createElement('span');
            chip.className = 'chip';
            chip.textContent = kw + ' ';
            
            const removeBtn = document.createElement('span');
            removeBtn.className = 'remove-chip-btn';
            removeBtn.innerHTML = '&times;';
            
            // Remove chip handler
            removeBtn.addEventListener('click', () => {
                selectedKeywords = selectedKeywords.filter(k => k !== kw);
                renderChips();
                checkModified();
            });
            
            chip.appendChild(removeBtn);
            chipsContainer.appendChild(chip);
        });
    }

    function filterKeywordDropdown() {
        const query = keywordSearchInput.value.toLowerCase().trim();
        
        // Filter options: in master list, matches search query, and not currently selected as chip
        const filtered = masterKeywords.filter(kw => {
            const notSelected = !selectedKeywords.includes(kw);
            const matchesQuery = kw.toLowerCase().includes(query);
            return notSelected && matchesQuery;
        });
        
        keywordDropdown.innerHTML = '';
        
        if (filtered.length === 0) {
            const noResults = document.createElement('div');
            noResults.className = 'keyword-option no-results';
            noResults.textContent = 'Không tìm thấy từ khóa';
            keywordDropdown.appendChild(noResults);
        } else {
            filtered.forEach(kw => {
                const opt = document.createElement('div');
                opt.className = 'keyword-option';
                opt.textContent = kw;
                
                // Mouse down listener to fire before input blur
                opt.addEventListener('mousedown', (e) => {
                    e.preventDefault(); // Prevents input losing focus immediately
                    selectedKeywords.push(kw);
                    keywordSearchInput.value = '';
                    keywordDropdown.style.display = 'none';
                    renderChips();
                    checkModified();
                });
                
                keywordDropdown.appendChild(opt);
            });
        }
        
        keywordDropdown.style.display = 'block';
    }

    async function fetchProgress() {
        try {
            const res = await fetch('/api/progress');
            const data = await res.json();
            totalCount = data.total;
            totalCountSpan.textContent = totalCount;
            reviewedCountSpan.textContent = data.reviewed;
            
            const percent = totalCount > 0 ? ((data.reviewed / totalCount) * 100).toFixed(1) : 0;
            progressPercentSpan.textContent = `${percent}%`;
            progressBar.style.width = `${percent}%`;
            return data;
        } catch (err) {
            console.error('Error fetching progress:', err);
            return { current_id: 1, total: 8704, reviewed: 0 };
        }
    }

    async function loadPoem(id) {
        if (id < 1) return;
        currentId = id;
        
        setLoadingState(true);
        keywordSearchInput.value = '';

        try {
            const res = await fetch(`/api/poem/${id}`);
            if (res.status === 404) {
                alert('Không tìm thấy bài thơ này!');
                setLoadingState(false);
                return;
            }
            const data = await res.json();
            
            isDeleted = data.is_deleted || false;

            // Populate text fields
            poemIdSpan.textContent = data.id;
            categorySelect.value = data.category || '(CA DAO)';
            bookSelect.value = data.book || 'TỪ ĐIỂN TỤC NGỮ THÀNH NGỮ CA DAO VIỆT NAM - Quyển Thượng';
            poemTextarea.value = data.poem || '';
            explanationTextarea.value = data.explanation || '';
            
            // Handle keywords string -> split into selected keywords array
            const kwStr = data.keyword || '';
            selectedKeywords = kwStr.split(',')
                                     .map(s => s.trim().toLowerCase())
                                     .filter(s => s.length > 0);
            renderChips();
            
            // Set type select value. If type is empty (unedited), default to first available option
            if (data.type) {
                ensureTypeInSelect(data.type);
                typeSelect.value = data.type;
            } else if (typeSelect.options.length > 0) {
                typeSelect.value = typeSelect.options[0].value;
            } else {
                typeSelect.value = '';
            }
            
            // Save original states
            originalData = {
                poem: data.poem || '',
                category: data.category || '(CA DAO)',
                explanation: data.explanation || '',
                book: data.book || 'TỪ ĐIỂN TỤC NGỮ THÀNH NGỮ CA DAO VIỆT NAM - Quyển Thượng',
                keyword: selectedKeywords.join(', '),
                type: typeSelect.value
            };
            
            // Set status indicators
            if (isDeleted) {
                statusIndicator.className = 'status-indicator deleted';
                statusText.textContent = 'Đã xóa';
                saveBtn.disabled = true;
                deleteBtn.textContent = 'Khôi phục';
                deleteBtn.className = 'btn btn-green';
            } else {
                deleteBtn.textContent = '🗑 Xóa bài này';
                deleteBtn.className = 'btn btn-red';
                saveBtn.disabled = false;
                
                if (data.is_reviewed) {
                    statusIndicator.className = 'status-indicator saved';
                    statusText.textContent = 'Đã lưu';
                } else {
                    statusIndicator.className = 'status-indicator';
                    statusText.textContent = 'Chưa chỉnh sửa';
                }
            }
            
            // Update counts next/prev labels
            prevIdText.textContent = id > 1 ? id - 1 : '-';
            nextIdText.textContent = id < totalCount ? id + 1 : '-';
            prevBtn.disabled = id <= 1;
            nextBtn.disabled = id >= totalCount && !isDeleted;

            // Trigger updates
            updateSyllables();
            
            // Scroll to top of editor card
            document.querySelector('.editor-card').scrollIntoView({ behavior: 'smooth', block: 'center' });

        } catch (err) {
            console.error('Error loading poem:', err);
            alert('Lỗi tải dữ liệu bài thơ!');
        } finally {
            setLoadingState(false);
        }
    }

    function ensureTypeInSelect(typeName) {
        const optionExists = Array.from(typeSelect.options).some(opt => opt.value === typeName);
        if (!optionExists && typeName) {
            const opt = document.createElement('option');
            opt.value = typeName;
            opt.textContent = typeName;
            typeSelect.appendChild(opt);
        }
    }

    async function saveCurrentPoem() {
        const tVal = typeSelect.value;
        if (!tVal) {
            alert('Vui lòng chọn hoặc thêm một Thể loại phụ (Type) cho bài thơ!');
            typeSelect.focus();
            return;
        }

        setLoadingState(true);
        const kwValJoined = selectedKeywords.join(', ');
        
        const payload = {
            poem: poemTextarea.value,
            category: categorySelect.value,
            explanation: explanationTextarea.value,
            book: bookSelect.value,
            keyword: kwValJoined,
            type: tVal
        };

        try {
            const res = await fetch(`/api/poem/${currentId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                originalData = { ...payload };
                statusIndicator.className = 'status-indicator saved';
                statusText.textContent = 'Đã lưu';
                
                await fetchProgress();

                // Navigate to next poem automatically if there is one
                if (currentId < totalCount) {
                    loadPoem(currentId + 1);
                } else {
                    alert('Lưu thành công bài thơ cuối cùng!');
                    setLoadingState(false);
                }
            } else {
                const errData = await res.json();
                alert(errData.error || 'Lỗi khi lưu bài thơ!');
                setLoadingState(false);
            }
        } catch (err) {
            console.error('Error saving poem:', err);
            alert('Lỗi kết nối khi lưu bài thơ!');
            setLoadingState(false);
        }
    }

    async function deleteCurrentPoem() {
        if (isDeleted) {
            // Restore functionality
            setLoadingState(true);
            try {
                const res = await fetch(`/api/poem/${currentId}/restore`, { method: 'POST' });
                if (res.ok) {
                    alert('Đã khôi phục bài thơ.');
                    loadPoem(currentId);
                } else {
                    alert('Lỗi khi khôi phục bài thơ.');
                    setLoadingState(false);
                }
            } catch (err) {
                console.error(err);
                alert('Lỗi kết nối.');
                setLoadingState(false);
            }
            return;
        }

        if (!confirm('Bạn có chắc chắn muốn xóa bài thơ này? Lựa chọn này sẽ loại bỏ bài thơ khỏi tệp xuất dữ liệu.')) {
            return;
        }

        setLoadingState(true);
        try {
            const res = await fetch(`/api/poem/${currentId}/delete`, { method: 'POST' });
            if (res.ok) {
                const data = await res.json();
                await fetchProgress();
                loadPoem(data.next_id);
            } else {
                alert('Lỗi khi xóa bài thơ!');
                setLoadingState(false);
            }
        } catch (err) {
            console.error('Error deleting poem:', err);
            alert('Lỗi kết nối khi xóa bài thơ!');
            setLoadingState(false);
        }
    }

    async function addNewPoem() {
        const isModified = checkModified();
        if (isModified && !confirm('Dữ liệu đang sửa đổi chưa được lưu. Bạn có chắc muốn chuyển bài không?')) {
            return;
        }

        setLoadingState(true);
        try {
            const res = await fetch('/api/poem/new', { method: 'POST' });
            if (res.ok) {
                const data = await res.json();
                await fetchProgress();
                loadPoem(data.id);
            } else {
                alert('Lỗi khi thêm bài mới!');
                setLoadingState(false);
            }
        } catch (err) {
            console.error('Error adding poem:', err);
            alert('Lỗi kết nối khi thêm bài mới!');
            setLoadingState(false);
        }
    }

    function navigateTo(id) {
        if (id < 1) return;
        
        const isModified = checkModified();
        if (isModified) {
            if (!confirm('Dữ liệu đang sửa đổi chưa được lưu. Bạn có chắc muốn chuyển bài không?')) {
                return;
            }
        }
        loadPoem(id);
    }

    function checkModified() {
        if (isDeleted) return false;
        
        const isPoemMod = poemTextarea.value !== originalData.poem;
        const isCategoryMod = categorySelect.value !== originalData.category;
        const isExpMod = explanationTextarea.value !== originalData.explanation;
        const isBookMod = bookSelect.value !== originalData.book;
        const isTypeMod = typeSelect.value !== originalData.type;
        
        // Sort arrays to check content modification order-independently
        const currentKwStr = selectedKeywords.slice().sort().join(', ');
        const origKwStr = originalData.keyword.split(',').map(s => s.trim().toLowerCase()).filter(s => s.length > 0).sort().join(', ');
        const isKeywordMod = currentKwStr !== origKwStr;
        
        const modified = isPoemMod || isCategoryMod || isExpMod || isBookMod || isKeywordMod || isTypeMod;
        
        if (modified) {
            statusIndicator.className = 'status-indicator modified';
            statusText.textContent = 'Đang chỉnh sửa';
        } else {
            const isSaved = statusIndicator.classList.contains('saved');
            statusIndicator.className = isSaved ? 'status-indicator saved' : 'status-indicator';
            statusText.textContent = isSaved ? 'Đã lưu' : 'Chưa chỉnh sửa';
        }
        return modified;
    }

    function updateSyllables() {
        const text = poemTextarea.value;
        if (!text.trim()) {
            syllableDisplay.innerHTML = '';
            return;
        }

        const lines = text.split('\n');
        syllableDisplay.innerHTML = '';

        lines.forEach(line => {
            const words = line.trim().split(/\s+/).filter(w => w.length > 0);
            const count = words.length;
            
            const badge = document.createElement('span');
            badge.className = 'syllable-line';
            badge.textContent = `${count} chữ`;
            
            if (count === 6) {
                badge.classList.add('perfect-6');
            } else if (count === 8) {
                badge.classList.add('perfect-8');
            }
            
            syllableDisplay.appendChild(badge);
        });
    }

    function setLoadingState(loading) {
        prevBtn.disabled = loading || currentId <= 1;
        nextBtn.disabled = loading || (currentId >= totalCount && !isDeleted);
        saveBtn.disabled = loading || isDeleted;
        deleteBtn.disabled = loading;
        addBtn.disabled = loading;
        jumpBtn.disabled = loading;
        jumpInput.disabled = loading;
    }
});
