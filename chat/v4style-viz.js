// ============= Visualization Cell Manager =============
// Creates visualization cells in the notebook interface

const VizCellManager = {
    cellCounter: 0,
    cells: new Map(),

    // Parse CSV data
    parseCSV(text) {
        const lines = text.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        const data = [];

        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            if (values.length === headers.length) {
                data.push(values);
            }
        }

        return { headers, data };
    },

    // Analyze column types
    analyzeColumn(values) {
        const numericValues = values.map(v => parseFloat(v)).filter(v => !isNaN(v));
        const isNumeric = numericValues.length > values.length * 0.8;

        if (isNumeric) {
            const min = Math.min(...numericValues);
            const max = Math.max(...numericValues);
            const mean = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
            const sorted = [...numericValues].sort((a, b) => a - b);
            const median = sorted[Math.floor(sorted.length / 2)];
            const variance = numericValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / numericValues.length;
            const stdDev = Math.sqrt(variance);

            return {
                type: 'numeric',
                min, max, mean, median, stdDev,
                uniqueCount: new Set(numericValues).size,
                values: numericValues
            };
        } else {
            const uniqueValues = [...new Set(values)];
            const valueCounts = {};
            values.forEach(v => valueCounts[v] = (valueCounts[v] || 0) + 1);

            return {
                type: 'categorical',
                uniqueValues,
                uniqueCount: uniqueValues.length,
                valueCounts,
                values
            };
        }
    },

    // Process CSV and create visualization cells
    createFromCSV(csvData, fileName) {
        const { headers, data } = this.parseCSV(csvData);
        const columnInfo = headers.map((header, idx) => {
            const columnValues = data.map(row => row[idx]);
            const analysis = this.analyzeColumn(columnValues);
            return {
                name: header,
                index: idx,
                ...analysis
            };
        });

        const numericColumns = columnInfo.filter(col => col.type === 'numeric');

        // Create data table cell FIRST
        this.createDataTableCell(data, columnInfo, fileName);

        // Create histogram cell
        this.createHistogramCell(data, columnInfo, fileName);

        // Scatter matrix disabled per user request
        // if (numericColumns.length >= 2) {
        //     this.createMatrixCell(data, columnInfo, fileName);
        // }

        // Scroll to first viz cell
        setTimeout(() => {
            const cells = document.querySelectorAll('.viz-cell');
            if (cells.length > 0) {
                cells[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }, 300);
    },

    // Create data table cell with pagination
    createDataTableCell(data, columnInfo, fileName) {
        const cellId = `viztable-${this.cellCounter++}`;
        const container = document.getElementById('contentContainer');
        const rowsPerPage = 20;

        const cellHTML = `
            <div class="viz-cell" id="${cellId}">
                <div class="viz-cell-toolbar">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="viz-cell-label">📋 Data Table - ${fileName}</span>
                        <span class="viz-privacy-badge" title="This visualization is rendered in your browser only. The AI assistant cannot see this data.">
                            🔒 Browser Only - AI doesn't see this
                        </span>
                    </div>
                    <div class="viz-cell-controls">
                        <span style="font-size: 11px; color: #858585;">${data.length} rows × ${columnInfo.length} columns</span>
                        <button class="viz-cell-btn" onclick="this.closest('.viz-cell').remove()">✕</button>
                    </div>
                </div>
                <div class="viz-table-container" id="${cellId}-container">
                    <table class="viz-table" id="${cellId}-table">
                        <thead>
                            <tr>
                                <th style="width: 40px; text-align: center;">#</th>
                                ${columnInfo.map(col => `
                                    <th>
                                        ${col.name}
                                        <span class="column-type-badge ${col.type === 'numeric' ? 'numeric' : ''}">${col.type}</span>
                                    </th>
                                `).join('')}
                            </tr>
                        </thead>
                        <tbody id="${cellId}-tbody">
                        </tbody>
                    </table>
                </div>
                <div class="viz-table-info">
                    <div>
                        Showing <span id="${cellId}-showing">1-${Math.min(rowsPerPage, data.length)}</span> of ${data.length} rows
                    </div>
                    <div class="viz-table-pagination">
                        <button id="${cellId}-first">⟨⟨ First</button>
                        <button id="${cellId}-prev">⟨ Prev</button>
                        <span id="${cellId}-page-info">Page 1 of ${Math.ceil(data.length / rowsPerPage)}</span>
                        <button id="${cellId}-next">Next ⟩</button>
                        <button id="${cellId}-last">Last ⟩⟩</button>
                    </div>
                </div>
            </div>
        `;

        container.insertAdjacentHTML('beforeend', cellHTML);

        // Pagination logic
        let currentPage = 0;
        const totalPages = Math.ceil(data.length / rowsPerPage);

        const renderPage = (page) => {
            const tbody = document.getElementById(`${cellId}-tbody`);
            const start = page * rowsPerPage;
            const end = Math.min(start + rowsPerPage, data.length);

            tbody.innerHTML = '';

            for (let i = start; i < end; i++) {
                const row = data[i];
                const tr = document.createElement('tr');

                // Row number
                const tdNum = document.createElement('td');
                tdNum.style.textAlign = 'center';
                tdNum.style.color = '#858585';
                tdNum.style.fontWeight = '600';
                tdNum.textContent = i + 1;
                tr.appendChild(tdNum);

                // Data cells
                row.forEach((cell, idx) => {
                    const td = document.createElement('td');
                    const col = columnInfo[idx];

                    if (col.type === 'numeric') {
                        const num = parseFloat(cell);
                        td.textContent = isNaN(num) ? cell : num.toFixed(2);
                        td.style.textAlign = 'right';
                        td.style.fontFamily = 'Consolas, Monaco, monospace';
                    } else {
                        td.textContent = cell;
                    }

                    tr.appendChild(td);
                });

                tbody.appendChild(tr);
            }

            // Update pagination info
            document.getElementById(`${cellId}-showing`).textContent = `${start + 1}-${end}`;
            document.getElementById(`${cellId}-page-info`).textContent = `Page ${page + 1} of ${totalPages}`;

            // Update button states
            document.getElementById(`${cellId}-first`).disabled = page === 0;
            document.getElementById(`${cellId}-prev`).disabled = page === 0;
            document.getElementById(`${cellId}-next`).disabled = page === totalPages - 1;
            document.getElementById(`${cellId}-last`).disabled = page === totalPages - 1;
        };

        // Event listeners
        document.getElementById(`${cellId}-first`).addEventListener('click', () => {
            currentPage = 0;
            renderPage(currentPage);
        });

        document.getElementById(`${cellId}-prev`).addEventListener('click', () => {
            if (currentPage > 0) {
                currentPage--;
                renderPage(currentPage);
            }
        });

        document.getElementById(`${cellId}-next`).addEventListener('click', () => {
            if (currentPage < totalPages - 1) {
                currentPage++;
                renderPage(currentPage);
            }
        });

        document.getElementById(`${cellId}-last`).addEventListener('click', () => {
            currentPage = totalPages - 1;
            renderPage(currentPage);
        });

        // Initial render
        renderPage(0);
    },

    // Create 3D visualization cell
    create3DCell(data, columnInfo, fileName) {
        const cellId = `viz3d-${this.cellCounter++}`;
        const container = document.getElementById('contentContainer');

        const numericColumns = columnInfo.filter(col => col.type === 'numeric');
        const categoricalColumns = columnInfo.filter(col => col.type === 'categorical');

        const cellHTML = `
            <div class="viz-cell" id="${cellId}">
                <div class="viz-cell-toolbar">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="viz-cell-label">📊 3D Scatter Plot - ${fileName}</span>
                        <span class="viz-privacy-badge" title="This visualization is rendered in your browser only. The AI assistant cannot see this data.">
                            🔒 Browser Only - AI doesn't see this
                        </span>
                    </div>
                    <div class="viz-cell-controls">
                        <label style="font-size: 10px; color: #858585; margin-right: 4px;">X:</label>
                        <select class="viz-cell-select" id="${cellId}-x"></select>
                        <label style="font-size: 10px; color: #858585; margin-right: 4px;">Y:</label>
                        <select class="viz-cell-select" id="${cellId}-y"></select>
                        <label style="font-size: 10px; color: #858585; margin-right: 4px;">Z:</label>
                        <select class="viz-cell-select" id="${cellId}-z"></select>
                        <label style="font-size: 10px; color: #858585; margin-left: 8px; margin-right: 4px;">Color:</label>
                        <select class="viz-cell-select" id="${cellId}-color"></select>
                        <button class="viz-cell-btn" id="${cellId}-play">▶ Play</button>
                        <button class="viz-cell-btn" id="${cellId}-reset">Reset</button>
                        <button class="viz-cell-btn" onclick="this.closest('.viz-cell').remove()">✕</button>
                    </div>
                </div>
                <div class="viz-cell-canvas" id="${cellId}-canvas">
                    <div class="viz-cell-info">
                        <div>Points: ${data.length}</div>
                        <div id="${cellId}-frame">Frame: 0 / ${data.length}</div>
                    </div>
                </div>
                <div class="viz-cell-legend" id="${cellId}-legend"></div>
            </div>
        `;

        container.insertAdjacentHTML('beforeend', cellHTML);

        // Populate dropdowns
        const xSelect = document.getElementById(`${cellId}-x`);
        const ySelect = document.getElementById(`${cellId}-y`);
        const zSelect = document.getElementById(`${cellId}-z`);
        const colorSelect = document.getElementById(`${cellId}-color`);

        numericColumns.forEach((col, idx) => {
            xSelect.add(new Option(col.name, col.index));
            ySelect.add(new Option(col.name, col.index));
            zSelect.add(new Option(col.name, col.index));
        });

        colorSelect.add(new Option('None', ''));
        categoricalColumns.forEach(col => {
            if (col.uniqueCount <= 20) {
                colorSelect.add(new Option(col.name, col.index));
            }
        });

        // Set default values
        xSelect.value = numericColumns[0].index;
        ySelect.value = numericColumns[1].index;
        zSelect.value = numericColumns[2].index;

        if (categoricalColumns.length > 0 && categoricalColumns[categoricalColumns.length - 1].uniqueCount <= 20) {
            colorSelect.value = categoricalColumns[categoricalColumns.length - 1].index;
        }

        // Initialize Three.js
        const cell = this.init3DCell(cellId, data, columnInfo);
        this.cells.set(cellId, cell);

        // Event listeners
        xSelect.addEventListener('change', () => cell.updateAxes());
        ySelect.addEventListener('change', () => cell.updateAxes());
        zSelect.addEventListener('change', () => cell.updateAxes());
        colorSelect.addEventListener('change', () => cell.updateColor());
        document.getElementById(`${cellId}-play`).addEventListener('click', () => cell.togglePlay());
        document.getElementById(`${cellId}-reset`).addEventListener('click', () => cell.reset());
    },

    // Initialize Three.js for a cell
    init3DCell(cellId, data, columnInfo) {
        const canvasContainer = document.getElementById(`${cellId}-canvas`);
        const width = canvasContainer.offsetWidth;
        const height = canvasContainer.offsetHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1e1e1e);

        const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        camera.position.set(6, 6, 6);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        canvasContainer.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(10, 10, 10);
        scene.add(directionalLight);

        // Create axes
        const axisLength = 8;
        const axisColors = [0x4ec9b0, 0x569cd6, 0xdcdcaa];

        for (let i = 0; i < 3; i++) {
            const material = new THREE.LineBasicMaterial({ color: axisColors[i] });
            const points = [];

            if (i === 0) {
                points.push(new THREE.Vector3(0, 0, 0));
                points.push(new THREE.Vector3(axisLength, 0, 0));
            } else if (i === 1) {
                points.push(new THREE.Vector3(0, 0, 0));
                points.push(new THREE.Vector3(0, axisLength, 0));
            } else {
                points.push(new THREE.Vector3(0, 0, 0));
                points.push(new THREE.Vector3(0, 0, axisLength));
            }

            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, material);
            scene.add(line);
        }

        const gridHelper = new THREE.GridHelper(16, 16, 0x3e3e42, 0x2d2d30);
        scene.add(gridHelper);

        const cell = {
            cellId,
            data,
            columnInfo,
            scene,
            camera,
            renderer,
            controls,
            dataPoints: [],
            isPlaying: false,
            currentFrame: 0,

            normalizeData(data, colIndex) {
                const values = data.map(row => parseFloat(row[colIndex])).filter(v => !isNaN(v));
                const min = Math.min(...values);
                const max = Math.max(...values);
                return values.map(v => ((v - min) / (max - min)) * 6 - 3);
            },

            generateColors(count) {
                const colors = [];
                const golden_ratio = 0.618033988749895;
                let hue = Math.random();

                for (let i = 0; i < count; i++) {
                    hue += golden_ratio;
                    hue %= 1;
                    colors.push(this.hslToRgb(hue, 0.7, 0.6));
                }

                return colors;
            },

            hslToRgb(h, s, l) {
                let r, g, b;

                if (s === 0) {
                    r = g = b = l;
                } else {
                    const hue2rgb = (p, q, t) => {
                        if (t < 0) t += 1;
                        if (t > 1) t -= 1;
                        if (t < 1/6) return p + (q - p) * 6 * t;
                        if (t < 1/2) return q;
                        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                        return p;
                    };

                    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
                    const p = 2 * l - q;

                    r = hue2rgb(p, q, h + 1/3);
                    g = hue2rgb(p, q, h);
                    b = hue2rgb(p, q, h - 1/3);
                }

                return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
            },

            createPoints() {
                this.dataPoints.forEach(point => this.scene.remove(point));
                this.dataPoints = [];

                const xIdx = parseInt(document.getElementById(`${this.cellId}-x`).value);
                const yIdx = parseInt(document.getElementById(`${this.cellId}-y`).value);
                const zIdx = parseInt(document.getElementById(`${this.cellId}-z`).value);
                const colorIdx = document.getElementById(`${this.cellId}-color`).value;

                const xNorm = this.normalizeData(this.data, xIdx);
                const yNorm = this.normalizeData(this.data, yIdx);
                const zNorm = this.normalizeData(this.data, zIdx);

                let colors = null;
                if (colorIdx !== '') {
                    const col = this.columnInfo[parseInt(colorIdx)];
                    if (col && col.type === 'categorical') {
                        const categoryColors = this.generateColors(col.uniqueValues.length);
                        colors = this.data.map(row => {
                            const categoryIdx = col.uniqueValues.indexOf(row[colorIdx]);
                            return categoryColors[categoryIdx];
                        });

                        // Update legend
                        const legend = document.getElementById(`${this.cellId}-legend`);
                        legend.innerHTML = col.uniqueValues.map((value, idx) => {
                            const color = `rgb(${categoryColors[idx][0]}, ${categoryColors[idx][1]}, ${categoryColors[idx][2]})`;
                            return `
                                <div class="viz-legend-item">
                                    <div class="viz-legend-color" style="background: ${color};"></div>
                                    <span>${value} (${col.valueCounts[value]})</span>
                                </div>
                            `;
                        }).join('');
                    }
                } else {
                    document.getElementById(`${this.cellId}-legend`).innerHTML = '';
                }

                for (let i = 0; i < this.data.length; i++) {
                    const geometry = new THREE.SphereGeometry(0.12, 16, 16);
                    const color = colors ? new THREE.Color(`rgb(${colors[i][0]}, ${colors[i][1]}, ${colors[i][2]})`) : new THREE.Color(0x4ec9b0);
                    const material = new THREE.MeshPhongMaterial({
                        color: color,
                        transparent: true,
                        opacity: 0
                    });

                    const sphere = new THREE.Mesh(geometry, material);
                    sphere.position.set(xNorm[i], yNorm[i], zNorm[i]);
                    this.scene.add(sphere);
                    this.dataPoints.push(sphere);
                }

                this.currentFrame = 0;
                document.getElementById(`${this.cellId}-frame`).textContent = `Frame: 0 / ${this.data.length}`;
            },

            updateAxes() {
                this.isPlaying = false;
                const btn = document.getElementById(`${this.cellId}-play`);
                btn.textContent = '▶ Play';
                this.createPoints();
            },

            updateColor() {
                this.isPlaying = false;
                const btn = document.getElementById(`${this.cellId}-play`);
                btn.textContent = '▶ Play';
                this.createPoints();
            },

            togglePlay() {
                this.isPlaying = !this.isPlaying;
                const btn = document.getElementById(`${this.cellId}-play`);
                btn.textContent = this.isPlaying ? '⏸ Pause' : '▶ Play';
            },

            reset() {
                this.isPlaying = false;
                this.currentFrame = 0;
                const btn = document.getElementById(`${this.cellId}-play`);
                btn.textContent = '▶ Play';
                this.dataPoints.forEach(point => {
                    point.material.opacity = 0;
                    point.scale.set(0.1, 0.1, 0.1);
                });
                document.getElementById(`${this.cellId}-frame`).textContent = `Frame: 0 / ${this.data.length}`;
            },

            animate() {
                requestAnimationFrame(() => this.animate());
                this.controls.update();

                if (this.isPlaying && this.currentFrame < this.data.length) {
                    const point = this.dataPoints[this.currentFrame];
                    point.material.opacity = 1;
                    point.scale.set(1, 1, 1);

                    setTimeout(() => {
                        point.scale.set(1.3, 1.3, 1.3);
                        setTimeout(() => point.scale.set(1, 1, 1), 100);
                    }, 50);

                    this.currentFrame++;
                    document.getElementById(`${this.cellId}-frame`).textContent = `Frame: ${this.currentFrame} / ${this.data.length}`;

                    if (this.currentFrame >= this.data.length) {
                        this.isPlaying = false;
                        document.getElementById(`${this.cellId}-play`).textContent = '▶ Play';
                    }
                }

                this.dataPoints.forEach(point => {
                    if (point.material.opacity > 0) {
                        point.rotation.y += 0.01;
                    }
                });

                this.renderer.render(this.scene, this.camera);
            }
        };

        cell.createPoints();
        cell.animate();

        return cell;
    },

    // Create histogram cell
    createHistogramCell(data, columnInfo, fileName) {
        const cellId = `vizhist-${this.cellCounter++}`;
        const container = document.getElementById('contentContainer');

        const cellHTML = `
            <div class="viz-cell" id="${cellId}">
                <div class="viz-cell-toolbar">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="viz-cell-label">📊 Histograms - ${fileName}</span>
                        <span class="viz-privacy-badge" title="This visualization is rendered in your browser only. The AI assistant cannot see this data.">
                            🔒 Browser Only - AI doesn't see this
                        </span>
                    </div>
                    <div class="viz-cell-controls">
                        <button class="viz-cell-btn" onclick="this.closest('.viz-cell').remove()">✕</button>
                    </div>
                </div>
                <div class="viz-histogram-container">
                    <div class="viz-histogram-grid" id="${cellId}-grid"></div>
                </div>
            </div>
        `;

        container.insertAdjacentHTML('beforeend', cellHTML);

        const grid = document.getElementById(`${cellId}-grid`);

        columnInfo.forEach(col => {
            const card = document.createElement('div');
            card.className = 'viz-histogram-card';

            const title = document.createElement('h4');
            title.textContent = col.name;
            card.appendChild(title);

            const stats = document.createElement('div');
            stats.className = 'stats';

            if (col.type === 'numeric') {
                stats.innerHTML = `
                    Min: ${col.min.toFixed(2)} | Max: ${col.max.toFixed(2)} | Mean: ${col.mean.toFixed(2)}
                `;
            } else {
                stats.innerHTML = `${col.uniqueCount} unique values`;
            }

            card.appendChild(stats);

            const canvas = document.createElement('canvas');
            card.appendChild(canvas);

            grid.appendChild(card);

            if (col.type === 'numeric') {
                this.drawNumericHistogram(canvas, col);
            } else {
                this.drawCategoricalHistogram(canvas, col);
            }
        });
    },

    drawNumericHistogram(canvas, col) {
        const ctx = canvas.getContext('2d');
        const bins = 20;
        const binCounts = new Array(bins).fill(0);
        const binWidth = (col.max - col.min) / bins;

        col.values.forEach(val => {
            const binIndex = Math.min(Math.floor((val - col.min) / binWidth), bins - 1);
            binCounts[binIndex]++;
        });

        const labels = [];
        for (let i = 0; i < bins; i++) {
            const binStart = col.min + i * binWidth;
            labels.push(binStart.toFixed(1));
        }

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Frequency',
                    data: binCounts,
                    backgroundColor: '#4ec9b0',
                    borderColor: '#3e3e42',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            font: { size: 10 },
                            color: '#858585'
                        },
                        grid: {
                            color: '#3e3e42'
                        }
                    },
                    x: {
                        ticks: {
                            font: { size: 8 },
                            color: '#858585',
                            maxRotation: 45,
                            minRotation: 45
                        },
                        grid: {
                            color: '#3e3e42'
                        }
                    }
                }
            }
        });
    },

    drawCategoricalHistogram(canvas, col) {
        const ctx = canvas.getContext('2d');
        const sortedCategories = Object.entries(col.valueCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 20);

        const labels = sortedCategories.map(e => e[0]);
        const counts = sortedCategories.map(e => e[1]);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Count',
                    data: counts,
                    backgroundColor: '#569cd6',
                    borderColor: '#3e3e42',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            font: { size: 10 },
                            color: '#858585'
                        },
                        grid: {
                            color: '#3e3e42'
                        }
                    },
                    x: {
                        ticks: {
                            font: { size: 9 },
                            color: '#858585',
                            maxRotation: 45,
                            minRotation: 45
                        },
                        grid: {
                            color: '#3e3e42'
                        }
                    }
                }
            }
        });
    },

    // Create scatter matrix cell
    createMatrixCell(data, columnInfo, fileName) {
        const cellId = `vizmatrix-${this.cellCounter++}`;
        const container = document.getElementById('contentContainer');

        const cellHTML = `
            <div class="viz-cell" id="${cellId}">
                <div class="viz-cell-toolbar">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="viz-cell-label">📊 Scatter Matrix - ${fileName}</span>
                        <span class="viz-privacy-badge" title="This visualization is rendered in your browser only. The AI assistant cannot see this data.">
                            🔒 Browser Only - AI doesn't see this
                        </span>
                    </div>
                    <div class="viz-cell-controls">
                        <button class="viz-cell-btn" onclick="this.closest('.viz-cell').remove()">✕</button>
                    </div>
                </div>
                <div class="viz-matrix-container">
                    <div class="viz-matrix-grid" id="${cellId}-grid"></div>
                </div>
            </div>
        `;

        container.insertAdjacentHTML('beforeend', cellHTML);

        const numericCols = columnInfo.filter(col => col.type === 'numeric');
        const grid = document.getElementById(`${cellId}-grid`);

        const cellSize = Math.min(120, Math.floor((window.innerWidth - 100) / numericCols.length));
        grid.style.gridTemplateColumns = `repeat(${numericCols.length}, ${cellSize}px)`;
        grid.style.gridTemplateRows = `repeat(${numericCols.length}, ${cellSize}px)`;

        for (let row = 0; row < numericCols.length; row++) {
            for (let col = 0; col < numericCols.length; col++) {
                const cell = document.createElement('div');
                cell.className = 'viz-matrix-cell';

                if (row === col) {
                    cell.classList.add('diagonal');
                    cell.textContent = numericCols[row].name;
                } else {
                    const canvas = document.createElement('canvas');
                    canvas.width = cellSize * 2;
                    canvas.height = cellSize * 2;
                    cell.appendChild(canvas);

                    this.drawScatterPlot(canvas, numericCols[col].values, numericCols[row].values);

                    // Make cell clickable
                    cell.style.cursor = 'pointer';
                    cell.title = `Click to enlarge: ${numericCols[col].name} vs ${numericCols[row].name}`;
                    cell.addEventListener('click', () => {
                        this.showScatterModal(
                            numericCols[col].values,
                            numericCols[row].values,
                            numericCols[col].name,
                            numericCols[row].name,
                            fileName
                        );
                    });
                }

                grid.appendChild(cell);
            }
        }
    },

    drawScatterPlot(canvas, xValues, yValues, pointSize = 2) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = 10;

        ctx.clearRect(0, 0, width, height);

        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);

        for (let i = 0; i < xValues.length; i++) {
            const x = padding + ((xValues[i] - xMin) / (xMax - xMin)) * (width - 2 * padding);
            const y = height - padding - ((yValues[i] - yMin) / (yMax - yMin)) * (height - 2 * padding);

            ctx.beginPath();
            ctx.arc(x, y, pointSize, 0, 2 * Math.PI);
            ctx.fillStyle = '#4ec9b0';
            ctx.fill();
        }
    },

    showScatterModal(xValues, yValues, xName, yName, fileName) {
        const modal = document.getElementById('scatterModal');
        const title = document.getElementById('scatterModalTitle');
        const canvasContainer = document.getElementById('scatterModalCanvas');
        const info = document.getElementById('scatterModalInfo');

        // Set title
        title.textContent = `${yName} vs ${xName} - ${fileName}`;

        // Clear previous canvas
        canvasContainer.innerHTML = '';

        // Create large canvas
        const canvas = document.createElement('canvas');
        const size = Math.min(800, window.innerWidth * 0.8);
        canvas.width = size;
        canvas.height = size;
        canvasContainer.appendChild(canvas);

        // Draw scatter plot with larger points
        this.drawScatterPlot(canvas, xValues, yValues, 4);

        // Calculate correlation
        const correlation = this.calculateCorrelation(xValues, yValues);

        // Show info
        const xMin = Math.min(...xValues).toFixed(2);
        const xMax = Math.max(...xValues).toFixed(2);
        const yMin = Math.min(...yValues).toFixed(2);
        const yMax = Math.max(...yValues).toFixed(2);

        info.innerHTML = `
            <div class="scatter-modal-info-item">
                <span class="scatter-modal-info-label">${xName} Range</span>
                <span>${xMin} - ${xMax}</span>
            </div>
            <div class="scatter-modal-info-item">
                <span class="scatter-modal-info-label">${yName} Range</span>
                <span>${yMin} - ${yMax}</span>
            </div>
            <div class="scatter-modal-info-item">
                <span class="scatter-modal-info-label">Correlation</span>
                <span>${correlation.toFixed(3)}</span>
            </div>
            <div class="scatter-modal-info-item">
                <span class="scatter-modal-info-label">Data Points</span>
                <span>${xValues.length}</span>
            </div>
        `;

        // Show modal
        modal.classList.add('active');
    },

    calculateCorrelation(xValues, yValues) {
        const n = xValues.length;
        const xMean = xValues.reduce((a, b) => a + b, 0) / n;
        const yMean = yValues.reduce((a, b) => a + b, 0) / n;

        let numerator = 0;
        let xDenominator = 0;
        let yDenominator = 0;

        for (let i = 0; i < n; i++) {
            const xDiff = xValues[i] - xMean;
            const yDiff = yValues[i] - yMean;
            numerator += xDiff * yDiff;
            xDenominator += xDiff * xDiff;
            yDenominator += yDiff * yDiff;
        }

        const denominator = Math.sqrt(xDenominator * yDenominator);
        return denominator === 0 ? 0 : numerator / denominator;
    }
};

// Global function to close scatter modal
function closeScatterModal() {
    const modal = document.getElementById('scatterModal');
    modal.classList.remove('active');
}

// Close modal when clicking outside
document.addEventListener('click', function(event) {
    const modal = document.getElementById('scatterModal');
    if (modal && event.target === modal) {
        closeScatterModal();
    }
});

// Auto-trigger on CSV upload
document.addEventListener('DOMContentLoaded', function() {
    const fileUpload = document.getElementById('fileUpload');
    const browserFileInput = document.getElementById('browserFileInput');

    if (fileUpload) {
        fileUpload.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file && file.name.endsWith('.csv')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const csvData = e.target.result;

                    // Call original handler if it exists
                    if (window.handleFileUpload) {
                        window.handleFileUpload(event);
                    }

                    // Create visualization cells - DISABLED (only show df.info())
                    // VizCellManager.createFromCSV(csvData, file.name);
                };
                reader.readAsText(file);
            } else if (window.handleFileUpload) {
                window.handleFileUpload(event);
            }
        });
    }

    if (browserFileInput) {
        browserFileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file && file.name.endsWith('.csv')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const csvData = e.target.result;

                    // Call original handler if it exists
                    if (window.handleBrowserFileUpload) {
                        window.handleBrowserFileUpload(event);
                    }

                    // Create visualization cells
                    VizCellManager.createFromCSV(csvData, file.name);
                };
                reader.readAsText(file);
            } else if (window.handleBrowserFileUpload) {
                window.handleBrowserFileUpload(event);
            }
        });
    }

    const notebookImport = document.getElementById('notebookImport');
    if (notebookImport) {
        notebookImport.addEventListener('change', function(event) {
            if (window.handleNotebookImport) {
                window.handleNotebookImport(event);
            }
        });
    }
});
