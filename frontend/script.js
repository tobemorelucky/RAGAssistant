const { createApp } = Vue;

createApp({
    data() {
        return {
            messages: [],
            userInput: '',
            isLoading: false,
            activeNav: 'newChat',
            abortController: null,
            sessionId: 'session_' + Date.now(),
            sessions: [],
            showHistorySidebar: false,
            isComposing: false,
            documents: [],
            documentsLoading: false,
            selectedFiles: [],
            isUploading: false,
            uploadProgress: '',
            uploadSteps: [],
            uploadProgressCollapsed: false,
            activeUploadJobId: '',
            uploadPollTimer: null,
            deleteJobs: {},
            deletePollTimers: {},
            deleteRemoveTimers: {},
            selectedDocumentFilenames: [],
            token: localStorage.getItem('accessToken') || '',
            currentUser: null,
            authMode: 'login',
            authForm: {
                username: '',
                password: '',
                role: 'user',
                admin_code: ''
            },
            authLoading: false
        };
    },
    computed: {
        isAuthenticated() {
            return !!this.token && !!this.currentUser;
        },
        isAdmin() {
            return this.currentUser?.role === 'admin';
        },
        selectedDocumentCount() {
            return this.selectedDocumentFilenames.length;
        },
        allSelectableDocumentsSelected() {
            const selectable = this.documents.filter(doc => !this.isDeleteActionLocked(doc.filename));
            return selectable.length > 0 && selectable.every(doc => this.selectedDocumentFilenames.includes(doc.filename));
        }
    },
    async mounted() {
        this.configureMarked();
        if (this.token) {
            try {
                await this.fetchMe();
            } catch (_) {
                this.handleLogout();
            }
        }
    },
    beforeUnmount() {
        this.stopUploadJobPolling();
        this.stopAllDeleteJobPolling();
        Object.values(this.deleteRemoveTimers).forEach(timer => clearTimeout(timer));
    },
    methods: {
        configureMarked() {
            marked.setOptions({
                highlight: function(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                },
                langPrefix: 'hljs language-',
                breaks: true,
                gfm: true
            });
        },

        parseMarkdown(text) {
            return marked.parse(text);
        },

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },

        authHeaders(extra = {}) {
            const headers = { ...extra };
            if (this.token) {
                headers.Authorization = `Bearer ${this.token}`;
            }
            return headers;
        },

        async authFetch(url, options = {}) {
            const opts = { ...options };
            opts.headers = this.authHeaders(opts.headers || {});
            const response = await fetch(url, opts);
            if (response.status === 401) {
                this.handleLogout();
                throw new Error('登录已过期，请重新登录');
            }
            return response;
        },

        async fetchMe() {
            const response = await this.authFetch('/auth/me');
            if (!response.ok) {
                throw new Error('璁よ瘉澶辫触');
            }
            this.currentUser = await response.json();
        },

        async handleAuthSubmit() {
            if (this.authLoading) return;
            const username = this.authForm.username.trim();
            const password = this.authForm.password.trim();
            if (!username || !password) {
                alert('鐢ㄦ埛鍚嶅拰瀵嗙爜涓嶈兘涓虹┖');
                return;
            }

            this.authLoading = true;
            try {
                const endpoint = this.authMode === 'login' ? '/auth/login' : '/auth/register';
                const payload = {
                    username,
                    password
                };
                if (this.authMode === 'register') {
                    payload.role = this.authForm.role;
                    payload.admin_code = this.authForm.admin_code || null;
                }

                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(data.detail || '璁よ瘉澶辫触');
                }

                this.token = data.access_token;
                this.currentUser = { username: data.username, role: data.role };
                localStorage.setItem('accessToken', this.token);
                this.authForm.password = '';
                this.authForm.admin_code = '';
                this.messages = [];
                this.sessionId = 'session_' + Date.now();
                this.activeNav = 'newChat';
            } catch (error) {
                alert(error.message);
            } finally {
                this.authLoading = false;
            }
        },

        handleLogout() {
            this.token = '';
            this.currentUser = null;
            this.messages = [];
            this.sessions = [];
            this.documents = [];
            this.selectedDocumentFilenames = [];
            this.activeNav = 'newChat';
            this.showHistorySidebar = false;
            localStorage.removeItem('accessToken');
        },

        handleCompositionStart() {
            this.isComposing = true;
        },

        handleCompositionEnd() {
            this.isComposing = false;
        },

        handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey && !this.isComposing) {
                event.preventDefault();
                this.handleSend();
            }
        },

        handleStop() {
            if (this.abortController) {
                this.abortController.abort();
            }
        },

        async handleSend() {
            if (!this.isAuthenticated) {
                alert('璇峰厛鐧诲綍');
                return;
            }

            const text = this.userInput.trim();
            if (!text || this.isLoading || this.isComposing) return;

            this.messages.push({
                text: text,
                isUser: true
            });

            this.userInput = '';
            this.$nextTick(() => {
                this.resetTextareaHeight();
                this.scrollToBottom();
            });

            this.isLoading = true;
            this.messages.push({
                text: '',
                isUser: false,
                isThinking: true,
                ragTrace: null,
                ragSteps: []
            });
            const botMsgIdx = this.messages.length - 1;

            this.abortController = new AbortController();

            try {
                const response = await this.authFetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: text,
                        session_id: this.sessionId
                    }),
                    signal: this.abortController.signal,
                });

                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);

                        if (eventStr.startsWith('data: ')) {
                            const dataStr = eventStr.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.type === 'content') {
                                    if (this.messages[botMsgIdx].isThinking) {
                                        this.messages[botMsgIdx].isThinking = false;
                                    }
                                    this.messages[botMsgIdx].text += data.content;
                                } else if (data.type === 'trace') {
                                    this.messages[botMsgIdx].ragTrace = data.rag_trace;
                                } else if (data.type === 'rag_step') {
                                    if (!this.messages[botMsgIdx].ragSteps) {
                                        this.messages[botMsgIdx].ragSteps = [];
                                    }
                                    this.messages[botMsgIdx].ragSteps.push(data.step);
                                } else if (data.type === 'error') {
                                    this.messages[botMsgIdx].isThinking = false;
                                    this.messages[botMsgIdx].text += `\n[Error: ${data.content}]`;
                                }
                            } catch (e) {
                                console.warn('SSE parse error:', e);
                            }
                        }
                    }
                    this.$nextTick(() => this.scrollToBottom());
                }

            } catch (error) {
                if (error.name === 'AbortError') {
                    this.messages[botMsgIdx].isThinking = false;
                    if (!this.messages[botMsgIdx].text) {
                        this.messages[botMsgIdx].text = '(宸茬粓姝㈠洖绛?';
                    } else {
                        this.messages[botMsgIdx].text += '\n\n_(鍥炵瓟宸茶缁堟)_';
                    }
                } else {
                    this.messages[botMsgIdx].isThinking = false;
                    this.messages[botMsgIdx].text = `鍠靛憸... 鍑轰簡鐐归棶棰橈細${error.message}`;
                }
            } finally {
                this.isLoading = false;
                this.abortController = null;
                this.$nextTick(() => this.scrollToBottom());
            }
        },

        autoResize(event) {
            const textarea = event.target;
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        },

        resetTextareaHeight() {
            if (this.$refs.textarea) {
                this.$refs.textarea.style.height = 'auto';
            }
        },

        scrollToBottom() {
            if (this.$refs.chatContainer) {
                this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
            }
        },

        handleNewChat() {
            if (!this.isAuthenticated) return;
            this.messages = [];
            this.sessionId = 'session_' + Date.now();
            this.activeNav = 'newChat';
            this.showHistorySidebar = false;
        },

        handleClearChat() {
            if (confirm('确定要清空当前对话吗？')) {
                this.messages = [];
            }
        },

        async handleHistory() {
            if (!this.isAuthenticated) return;
            this.activeNav = 'history';
            this.showHistorySidebar = true;
            try {
                const response = await this.authFetch('/sessions');
                if (!response.ok) {
                    throw new Error('Failed to load sessions');
                }
                const data = await response.json();
                this.sessions = data.sessions;
            } catch (error) {
                alert('加载历史记录失败：' + error.message);
            }
        },

        async loadSession(sessionId) {
            this.sessionId = sessionId;
            this.showHistorySidebar = false;
            this.activeNav = 'newChat';

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`);
                if (!response.ok) {
                    throw new Error('Failed to load session messages');
                }
                const data = await response.json();
                this.messages = data.messages.map(msg => ({
                    text: msg.content,
                    isUser: msg.type === 'human',
                    ragTrace: msg.rag_trace || null
                }));

                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            } catch (error) {
                alert('加载会话失败：' + error.message);
                this.messages = [];
            }
        },

        async deleteSession(sessionId) {
            if (!confirm(`纭畾瑕佸垹闄や細璇?"${sessionId}" 鍚楋紵`)) {
                return;
            }

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`, {
                    method: 'DELETE'
                });

                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.detail || 'Delete failed');
                }

                this.sessions = this.sessions.filter(s => s.session_id !== sessionId);

                if (this.sessionId === sessionId) {
                    this.messages = [];
                    this.sessionId = 'session_' + Date.now();
                    this.activeNav = 'newChat';
                }

                if (payload.message) {
                    alert(payload.message);
                }
            } catch (error) {
                alert('删除会话失败：' + error.message);
            }
        },

        handleSettings() {
            if (!this.isAdmin) {
                alert('仅管理员可访问文档管理');
                return;
            }
            this.activeNav = 'settings';
            this.showHistorySidebar = false;
            this.loadDocuments();
        },

        mergeDocumentsWithActiveDeletes(nextDocuments) {
            const merged = Array.isArray(nextDocuments) ? [...nextDocuments] : [];
            Object.keys(this.deleteJobs).forEach(filename => {
                const job = this.deleteJobs[filename];
                if (!job || job.status === 'failed') return;
                const exists = merged.some(doc => doc.filename === filename);
                if (!exists) {
                    const currentDoc = this.documents.find(doc => doc.filename === filename);
                    if (currentDoc) {
                        merged.push(currentDoc);
                    }
                }
            });
            return merged;
        },

        pruneSelectedDocuments() {
            const available = new Set(this.documents.map(doc => doc.filename));
            this.selectedDocumentFilenames = this.selectedDocumentFilenames.filter(filename => available.has(filename));
        },

        async loadDocuments() {
            this.documentsLoading = true;
            try {
                const response = await this.authFetch('/documents');
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || 'Failed to load documents');
                }
                const data = await response.json();
                this.documents = this.mergeDocumentsWithActiveDeletes(data.documents);
                this.pruneSelectedDocuments();
            } catch (error) {
                alert('加载文档列表失败：' + error.message);
            } finally {
                this.documentsLoading = false;
            }
        },

        isDocumentSelected(filename) {
            return this.selectedDocumentFilenames.includes(filename);
        },

        toggleDocumentSelection(filename) {
            if (this.isDeleteActionLocked(filename)) return;
            if (this.isDocumentSelected(filename)) {
                this.selectedDocumentFilenames = this.selectedDocumentFilenames.filter(item => item !== filename);
            } else {
                this.selectedDocumentFilenames = [...this.selectedDocumentFilenames, filename];
            }
        },

        toggleAllDocumentsSelection() {
            const selectable = this.documents
                .filter(doc => !this.isDeleteActionLocked(doc.filename))
                .map(doc => doc.filename);
            if (!selectable.length) {
                this.selectedDocumentFilenames = [];
                return;
            }
            this.selectedDocumentFilenames = this.allSelectableDocumentsSelected ? [] : selectable;
        },

        handleFileSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                const nextFiles = Array.from(files);
                const merged = [...this.selectedFiles];
                for (const file of nextFiles) {
                    const exists = merged.some(item =>
                        item.name === file.name &&
                        item.size === file.size &&
                        item.lastModified === file.lastModified
                    );
                    if (!exists) {
                        merged.push(file);
                    }
                }
                this.selectedFiles = merged;
                this.uploadProgress = '';
                this.uploadSteps = this.createUploadSteps();
                this.uploadProgressCollapsed = false;
                this.activeUploadJobId = '';
                if (this.$refs.fileInput) {
                    this.$refs.fileInput.value = '';
                }
            }
        },

        createUploadSteps() {
            return [
                { key: 'upload', label: '鏂囨。涓婁紶', percent: 0, status: 'pending', message: '' },
                { key: 'cleanup', label: '清理旧版本', percent: 0, status: 'pending', message: '' },
                { key: 'parse', label: '解析与分块', percent: 0, status: 'pending', message: '' },
                { key: 'parent_store', label: '鐖剁骇鍒嗗潡鍏ュ簱', percent: 0, status: 'pending', message: '' },
                { key: 'vector_store', label: '向量化入库', percent: 0, status: 'pending', message: '' },
            ];
        },

        updateUploadStep(key, percent, status = 'running', message = '') {
            if (!this.uploadSteps.length) {
                this.uploadSteps = this.createUploadSteps();
            }
            const idx = this.uploadSteps.findIndex(step => step.key === key);
            if (idx === -1) return;
            this.uploadSteps[idx] = {
                ...this.uploadSteps[idx],
                percent: Math.max(0, Math.min(100, Math.round(percent || 0))),
                status,
                message
            };
        },

        uploadFileWithProgress(file) {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                const formData = new FormData();
                formData.append('file', file);

                xhr.open('POST', '/documents/upload/async');
                const headers = this.authHeaders();
                Object.entries(headers).forEach(([key, value]) => xhr.setRequestHeader(key, value));

                xhr.upload.onprogress = (event) => {
                    if (!event.lengthComputable) return;
                    const percent = Math.round((event.loaded / event.total) * 100);
                    this.updateUploadStep('upload', percent, 'running', `宸蹭笂浼?${percent}%`);
                };

                xhr.onload = () => {
                    if (xhr.status === 401) {
                        this.handleLogout();
                        reject(new Error('登录已过期，请重新登录'));
                        return;
                    }

                    let data = {};
                    try {
                        data = JSON.parse(xhr.responseText || '{}');
                    } catch (e) {
                        reject(new Error('涓婁紶鍝嶅簲瑙ｆ瀽澶辫触'));
                        return;
                    }

                    if (xhr.status < 200 || xhr.status >= 300) {
                        reject(new Error(data.detail || `HTTP ${xhr.status}`));
                        return;
                    }

                    this.updateUploadStep('upload', 100, 'completed', '鏂囨。涓婁紶瀹屾垚');
                    resolve(data);
                };

                xhr.onerror = () => reject(new Error('涓婁紶璇锋眰澶辫触'));
                xhr.onabort = () => reject(new Error('上传已取消'));
                xhr.send(formData);
            });
        },

        syncUploadJob(job) {
            this.activeUploadJobId = job.job_id;
            this.uploadProgress = job.message || '';
            if (Array.isArray(job.steps)) {
                this.uploadSteps = job.steps.map(step => ({
                    key: step.key,
                    label: step.label,
                    percent: step.percent,
                    status: step.status,
                    message: step.message || ''
                }));
            }
            // 鍏ュ簱鎴愬姛鍚庤嚜鍔ㄦ敹璧锋楠ゆ槑缁嗭紝淇濈暀鎽樿渚涚敤鎴峰啀娆″睍寮€鏌ョ湅銆?
            if (job.status === 'completed') {
                this.uploadProgressCollapsed = true;
            }
        },

        toggleUploadProgressCollapsed() {
            this.uploadProgressCollapsed = !this.uploadProgressCollapsed;
        },

        stopUploadJobPolling() {
            if (this.uploadPollTimer) {
                clearInterval(this.uploadPollTimer);
                this.uploadPollTimer = null;
            }
        },

        resetSelectedFiles() {
            this.selectedFiles = [];
            if (this.$refs.fileInput) {
                this.$refs.fileInput.value = '';
            }
        },

        sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        },

        async waitForUploadJob(jobId, fileName, index, total) {
            while (true) {
                const response = await this.authFetch(`/documents/upload/jobs/${encodeURIComponent(jobId)}`);
                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || 'Failed to load upload job');
                }

                const job = await response.json();
                this.syncUploadJob(job);
                this.uploadProgress = `(${index}/${total}) ${fileName}: ${job.message || ''}`;

                if (job.status === 'completed') {
                    this.uploadProgressCollapsed = true;
                    return job;
                }
                if (job.status === 'failed') {
                    throw new Error(job.error || job.message || 'Upload job failed');
                }

                await this.sleep(1000);
            }
        },

        async uploadSelectedFiles() {
            if (!this.selectedFiles.length) {
                alert('璇峰厛閫夋嫨鏂囦欢');
                return;
            }

            this.isUploading = true;
            this.uploadProgressCollapsed = false;

            const total = this.selectedFiles.length;
            const successFiles = [];
            const failedFiles = [];

            try {
                for (let index = 0; index < total; index += 1) {
                    const file = this.selectedFiles[index];
                    this.uploadSteps = this.createUploadSteps();
                    this.activeUploadJobId = '';
                    this.uploadProgress = `(${index + 1}/${total}) 姝ｅ湪涓婁紶 ${file.name}`;
                    this.updateUploadStep('upload', 0, 'running', `鍑嗗涓婁紶 ${file.name}`);

                    try {
                        const data = await this.uploadFileWithProgress(file);
                        this.activeUploadJobId = data.job_id;
                        await this.waitForUploadJob(data.job_id, file.name, index + 1, total);
                        successFiles.push(file.name);
                    } catch (error) {
                        failedFiles.push(`${file.name}: ${error.message}`);
                        this.updateUploadStep('upload', 100, 'failed', error.message);
                    }
                }

                await this.loadDocuments();

                if (failedFiles.length === 0) {
                    this.uploadProgress = '批量上传完成，共 ' + successFiles.length + ' 个文件';
                } else {
                    this.uploadProgress = '批量上传完成，成功 ' + successFiles.length + ' 个，失败 ' + failedFiles.length + ' 个';
                    alert('部分文件上传失败：\\n' + failedFiles.join('\\n'));
                }
            } finally {
                this.isUploading = false;
                this.resetSelectedFiles();
            }
        },

        startUploadJobPolling(jobId) {
            this.stopUploadJobPolling();

            const poll = async () => {
                try {
                    const response = await this.authFetch(`/documents/upload/jobs/${encodeURIComponent(jobId)}`);
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error.detail || 'Failed to load upload job');
                    }

                    const job = await response.json();
                    this.syncUploadJob(job);

                    if (job.status === 'completed') {
                        this.stopUploadJobPolling();
                        this.isUploading = false;
                        this.resetSelectedFiles();
                        await this.loadDocuments();
                    } else if (job.status === 'failed') {
                        this.stopUploadJobPolling();
                        this.isUploading = false;
                    }
                } catch (error) {
                    this.uploadProgress = '进度查询失败：' + error.message;
                    this.stopUploadJobPolling();
                    this.isUploading = false;
                }
            };

            poll();
            this.uploadPollTimer = setInterval(poll, 1000);
        },

        async uploadDocument() {
            if (!this.selectedFiles.length) {
                alert('璇峰厛閫夋嫨鏂囦欢');
                return;
            }

            this.isUploading = true;
            this.uploadProgress = '姝ｅ湪涓婁紶...';
            this.uploadSteps = this.createUploadSteps();
            this.uploadProgressCollapsed = false;
            this.updateUploadStep('upload', 0, 'running', '鍑嗗涓婁紶');

            try {
                const data = await this.uploadFileWithProgress(this.selectedFiles[0]);
                this.uploadProgress = data.message;
                this.activeUploadJobId = data.job_id;
                this.startUploadJobPolling(data.job_id);
            } catch (error) {
                this.updateUploadStep('upload', 100, 'failed', error.message);
                this.uploadProgress = '上传失败：' + error.message;
                this.isUploading = false;
            }
        },

        createDeleteSteps() {
            return [
                { key: 'prepare', label: '鍑嗗鍒犻櫎', percent: 0, status: 'pending', message: '' },
                { key: 'bm25', label: '鍚屾 BM25 缁熻', percent: 0, status: 'pending', message: '' },
                { key: 'milvus', label: '鍒犻櫎鍚戦噺鏁版嵁', percent: 0, status: 'pending', message: '' },
                { key: 'parent_store', label: '鍒犻櫎鐖剁骇鍒嗗潡', percent: 0, status: 'pending', message: '' },
            ];
        },

        isDeletingDocument(filename) {
            const job = this.deleteJobs[filename];
            return job && job.status === 'running';
        },

        isDeleteActionLocked(filename) {
            const job = this.deleteJobs[filename];
            return job && (job.status === 'running' || job.status === 'completed');
        },

        getDeleteButtonIcon(filename) {
            const job = this.deleteJobs[filename];
            if (job?.status === 'running') return 'fas fa-spinner fa-spin';
            if (job?.status === 'completed') return 'fas fa-check';
            return 'fas fa-trash';
        },

        setDeleteJob(filename, nextJob) {
            this.deleteJobs = {
                ...this.deleteJobs,
                [filename]: {
                    ...(this.deleteJobs[filename] || {}),
                    ...nextJob
                }
            };
        },

        syncDeleteJob(filename, job) {
            const current = this.deleteJobs[filename] || {};
            // 鍚庣杩斿洖缁熶竴鐨勬楠ょ粨鏋勶紝鍓嶇鍙礋璐ｅ悓姝ュ埌褰撳墠鏂囨。琛屽唴鍗＄墖銆?
            this.setDeleteJob(filename, {
                jobId: job.job_id,
                status: job.status,
                message: job.message || '',
                collapsed: job.status === 'completed' ? true : Boolean(current.collapsed),
                steps: Array.isArray(job.steps) ? job.steps.map(step => ({
                    key: step.key,
                    label: step.label,
                    percent: step.percent,
                    status: step.status,
                    message: step.message || ''
                })) : this.createDeleteSteps()
            });
        },

        toggleDeleteJobCollapsed(filename) {
            const job = this.deleteJobs[filename];
            if (!job) return;
            this.setDeleteJob(filename, { collapsed: !job.collapsed });
        },

        stopDeleteJobPolling(filename) {
            const timer = this.deletePollTimers[filename];
            if (!timer) return;
            clearInterval(timer);
            const { [filename]: _removed, ...rest } = this.deletePollTimers;
            this.deletePollTimers = rest;
        },

        stopAllDeleteJobPolling() {
            Object.keys(this.deletePollTimers).forEach(filename => this.stopDeleteJobPolling(filename));
        },

        clearDeleteRemovalTimer(filename) {
            const timer = this.deleteRemoveTimers[filename];
            if (!timer) return;
            clearTimeout(timer);
            const { [filename]: _removed, ...rest } = this.deleteRemoveTimers;
            this.deleteRemoveTimers = rest;
        },

        scheduleDeletedDocumentRemoval(filename) {
            this.clearDeleteRemovalTimer(filename);
            // 鍒犻櫎瀹屾垚鍚庡厛淇濈暀 3 绉掓憳瑕侊紝鍐嶄粠褰撳墠鍒楄〃绉婚櫎骞跺埛鏂板悗绔姸鎬併€?
            const timer = setTimeout(async () => {
                this.documents = this.documents.filter(doc => doc.filename !== filename);
                this.selectedDocumentFilenames = this.selectedDocumentFilenames.filter(item => item !== filename);
                const { [filename]: _job, ...jobs } = this.deleteJobs;
                const { [filename]: _timer, ...timers } = this.deleteRemoveTimers;
                this.deleteJobs = jobs;
                this.deleteRemoveTimers = timers;
                await this.loadDocuments();
            }, 3000);
            this.deleteRemoveTimers = {
                ...this.deleteRemoveTimers,
                [filename]: timer
            };
        },

        startDeleteJobPolling(filename, jobId) {
            this.stopDeleteJobPolling(filename);

            const poll = async () => {
                try {
                    const response = await this.authFetch(`/documents/delete/jobs/${encodeURIComponent(jobId)}`);
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error.detail || 'Failed to load delete job');
                    }

                    const job = await response.json();
                    this.syncDeleteJob(filename, job);

                    if (job.status === 'completed') {
                        this.stopDeleteJobPolling(filename);
                        this.scheduleDeletedDocumentRemoval(filename);
                    } else if (job.status === 'failed') {
                        this.stopDeleteJobPolling(filename);
                    }
                } catch (error) {
                    this.setDeleteJob(filename, {
                        status: 'failed',
                        message: '删除进度查询失败：' + error.message,
                        collapsed: false,
                        steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps()
                    });
                    this.stopDeleteJobPolling(filename);
                }
            };

            poll();
            this.deletePollTimers = {
                ...this.deletePollTimers,
                [filename]: setInterval(poll, 1000)
            };
        },
        async deleteSelectedDocuments() {
            const filenames = this.selectedDocumentFilenames.filter(filename => !this.isDeleteActionLocked(filename));
            if (!filenames.length) {
                alert('请先选择要删除的文档');
                return;
            }
            if (!confirm('确定要批量删除 ' + filenames.length + ' 个文档吗？这将同时删除 Milvus 中的相关向量数据。')) {
                return;
            }

            filenames.forEach(filename => {
                this.clearDeleteRemovalTimer(filename);
                this.setDeleteJob(filename, {
                    status: 'running',
                    message: '正在提交批量删除任务...',
                    collapsed: false,
                    steps: this.createDeleteSteps().map(step => (
                        step.key === 'prepare'
                            ? { ...step, percent: 1, status: 'running', message: '正在提交删除任务' }
                            : step
                    ))
                });
            });

            try {
                const response = await this.authFetch('/documents/delete/async/batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filenames })
                });

                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || 'Batch delete failed');
                }

                const data = await response.json();
                (data.jobs || []).forEach(job => {
                    this.setDeleteJob(job.filename, {
                        jobId: job.job_id,
                        status: 'running',
                        message: job.message || ('正在删除 ' + job.filename),
                        collapsed: false
                    });
                    this.startDeleteJobPolling(job.filename, job.job_id);
                });
                this.selectedDocumentFilenames = [];
            } catch (error) {
                filenames.forEach(filename => {
                    this.setDeleteJob(filename, {
                        status: 'failed',
                        message: '批量删除失败：' + error.message,
                        collapsed: false,
                        steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps()
                    });
                });
            }
        },

        async deleteDocument(filename) {
            if (this.isDeletingDocument(filename)) {
                return;
            }
            if (!confirm('确定要删除文档 "' + filename + '" 吗？这将同时删除 Milvus 中的所有相关向量。')) {
                return;
            }

            this.clearDeleteRemovalTimer(filename);
            this.setDeleteJob(filename, {
                status: 'running',
                message: '正在提交删除任务...',
                collapsed: false,
                steps: this.createDeleteSteps().map(step => (
                    step.key === 'prepare'
                        ? { ...step, percent: 1, status: 'running', message: '正在提交删除任务' }
                        : step
                ))
            });

            try {
                const response = await this.authFetch(`/documents/delete/async/${encodeURIComponent(filename)}`, {
                    method: 'DELETE'
                });

                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || 'Delete failed');
                }

                const data = await response.json();
                this.setDeleteJob(filename, {
                    jobId: data.job_id,
                    status: 'running',
                    message: data.message || `姝ｅ湪鍒犻櫎 ${filename}`,
                    collapsed: false
                });
                this.startDeleteJobPolling(filename, data.job_id);

            } catch (error) {
                this.setDeleteJob(filename, {
                    status: 'failed',
                    message: '删除文档失败：' + error.message,
                    collapsed: false,
                    steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps()
                });
            }
        },

        getFileIcon(fileType) {
            if (fileType === 'PDF') {
                return 'fas fa-file-pdf';
            } else if (fileType === 'Word') {
                return 'fas fa-file-word';
            } else if (fileType === 'Excel') {
                return 'fas fa-file-excel';
            } else if (fileType === 'Text') {
                return 'fas fa-file-lines';
            } else if (fileType === 'Markdown') {
                return 'fab fa-markdown';
            } else if (fileType === 'CSV') {
                return 'fas fa-file-csv';
            }
            return 'fas fa-file';
        }
    },
    watch: {
        messages: {
            handler() {
                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            },
            deep: true
        }
    }
}).mount('#app');

