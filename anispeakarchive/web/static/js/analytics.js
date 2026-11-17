let charts = {};

function destroyCharts() {
    Object.values(charts).forEach(ch => {
        if (ch && typeof ch.destroy === 'function') ch.destroy();
    });
    charts = {};
}

Chart.defaults.font.family = 'Inter, sans-serif';
Chart.defaults.color = '#64748b';
const colors = ['#4f46e5', '#7c3aed', '#ec4899', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#f43f5e', '#14b8a6', '#84cc16'];


function capitalizeFirst(s) {
    if (s === null || s === undefined) return '';
    const str = String(s);
    if (str.length === 0) return str;
    return str.charAt(0).toUpperCase() + str.slice(1);
}


function renderComplexProfileTable(data, years, type) {
    if (!data || data.length === 0) return '<div class="muted">No data available.</div>';

    let thYears = years.map(y => `<th style="text-align:center; font-size: 0.75rem;">${y}</th>`).join('');

    let rows = data.map(row => {
        let tdYears = years.map(y => `<td style="text-align:center; font-weight:600; font-size:0.85rem; color:${row.years[y] > 0 ? 'var(--text)' : 'var(--border)'}">${row.years[y] || 0}</td>`).join('');

        // Build the metadata tags
        let tagsHtml = "";
        if (row.top_institutions && row.top_institutions.length) {
            tagsHtml += row.top_institutions.map(t => `<span class="micro-tag">${capitalizeFirst(t)}</span>`).join('');
        }
        if (row.top_authors && row.top_authors.length) {
            tagsHtml += row.top_authors.map(t => `<span class="micro-tag">${capitalizeFirst(t)}</span>`).join('');
        }
        if (row.top_venues && row.top_venues.length) {
            tagsHtml += row.top_venues.map(t => `<span class="micro-tag venue">${capitalizeFirst(t)}</span>`).join('');
        }
        if (row.top_species && row.top_species.length) {
            tagsHtml += row.top_species.map(t => `<span class="micro-tag species">${capitalizeFirst(t)}</span>`).join('');
        }

        return `
            <tr>
                <td style="min-width: 250px;">
                    <div style="font-weight:700; color:var(--text); margin-bottom: 4px;">${row.name}</div>
                    <div style="display:flex; flex-wrap:wrap; gap:2px;">${tagsHtml}</div>
                </td>
                ${tdYears}
                <td style="text-align:right; color:var(--accent); font-weight:800; font-size:1rem;">${row.total}</td>
            </tr>
        `;
    }).join('');

    return `
        <div class="card matrix-card" style="padding: 0; overflow: hidden;">
            <div class="table-wrapper">
                <table class="simple-table">
                    <thead>
                        <tr>
                            <th>${type === 'author' ? 'Author & Focus Areas' : 'Institution & Top Associations'}</th>
                            ${thYears}
                            <th style="text-align:right;">Total</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>
    `;
}

function renderLeaderboard(title, dataArray) {
    if (!dataArray || dataArray.length === 0) return '';
    const rows = dataArray.map(item => `
        <tr>
            <td>${capitalizeFirst(item.name)}</td>
            <td style="text-align: right; font-weight: 700; color: var(--text);">${item.count}</td>
        </tr>
    `).join('');

    return `
        <div class="card" style="padding: 0; overflow: hidden;">
            <h3 style="margin: 15px 15px 10px;">${title}</h3>
            <div class="table-wrapper">
                <table class="simple-table">
                    <thead><tr><th>Name</th><th style="text-align: right;">Count</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>
    `;
}

async function loadAnalytics() {
    try {
        const resp = await ((window.asaFetch && typeof window.asaFetch === "function") ? window.asaFetch : fetch)('/api/analytics/extensive');
        const data = await resp.json();

        // Legend for tag colors used in the tables below
        const legendEl = document.getElementById('matrixLegend');
        if (legendEl) {
            legendEl.innerHTML = `
              <span style="display:inline-flex;align-items:center;gap:10px;flex-wrap:wrap;">
                <span style="display:inline-flex;align-items:center;gap:6px;">
                  <span style="width:12px;height:12px;border-radius:3px;background:rgba(79,70,229,0.25);border:1px solid rgba(79,70,229,0.35);display:inline-block;"></span>
                  <span>Institutions / Authors</span>
                </span>
                <span style="display:inline-flex;align-items:center;gap:6px;">
                  <span style="width:12px;height:12px;border-radius:3px;background:#fef3c7;border:1px solid #fde68a;display:inline-block;"></span>
                  <span>Venues</span>
                </span>
                <span style="display:inline-flex;align-items:center;gap:6px;">
                  <span style="width:12px;height:12px;border-radius:3px;background:#dcfce7;border:1px solid #bbf7d0;display:inline-block;"></span>
                  <span>Species</span>
                </span>
              </span>
            `;
        }

        // 1. Update KPIs
        document.getElementById('kpiPapers').textContent = (data.overview.papers_total || 0).toLocaleString();
        document.getElementById('kpiAuthors').textContent = (data.overview.authors_total || 0).toLocaleString();
        document.getElementById('kpiSpecies').textContent = (data.overview.species_total || 0).toLocaleString();
        document.getElementById('kpiAvgAuthors').textContent = data.overview.avg_authors_per_paper || "-";

        destroyCharts();

        // 2. Timeline Line Chart
        const tlCtx = document.getElementById('timelineChart').getContext('2d');
        charts.timeline = new Chart(tlCtx, {
            type: 'line',
            data: {
                labels: data.timeline.map(d => d.year),
                datasets: [{
                    label: 'Publications',
                    data: data.timeline.map(d => d.count),
                    borderColor: '#4f46e5',
                    backgroundColor: 'rgba(79, 70, 229, 0.1)',
                    borderWidth: 2, fill: true, tension: 0.3, pointRadius: 1, pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { grid: { display: false } }, y: { beginAtZero: true } }
            }
        });

        // 3. Venues Bar Chart (responsive + long-label friendly)
        if (data.leaderboards.venues && data.leaderboards.venues.length) {
            const canvas = document.getElementById('venuesBarChart');
            const vCtx = canvas.getContext('2d');
            const venueFull = data.leaderboards.venues.map(d => d.name);

            charts.venues = new Chart(vCtx, {
                type: 'bar',
                data: {
                    labels: venueFull, // keep full labels; truncate in ticks callback
                    datasets: [{
                        label: 'Papers',
                        data: data.leaderboards.venues.map(d => d.count),
                        backgroundColor: 'rgba(245, 158, 11, 0.8)',
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                // Show full venue name in tooltip
                                title: function(items) {
                                    if (!items || !items.length) return '';
                                    const i = items[0].dataIndex;
                                    return venueFull[i] || '';
                                }
                            }
                        }
                    },
                    layout: {
                        padding: { left: 6, right: 10, top: 6, bottom: 6 }
                    },
                    scales: {
                        y: {
                            ticks: {
                                autoSkip: false,
                                font: function(ctx) {
                                    const w = (ctx && ctx.chart && ctx.chart.width) ? ctx.chart.width : 1000;
                                    return { size: w < 520 ? 10 : 12, weight: '600' };
                                },
                                callback: function(value) {
                                    const label = this.getLabelForValue(value) || '';
                                    const w = (this.chart && this.chart.width) ? this.chart.width : 1000;
                                    const max = (w < 520) ? 18 : 28;
                                    return label.length > max ? (label.slice(0, Math.max(0, max - 1)) + '…') : label;
                                }
                            },
                            grid: { color: 'rgba(15,23,42,.06)' }
                        },
                        x: {
                            beginAtZero: true,
                            ticks: {
                                font: function(ctx) {
                                    const w = (ctx && ctx.chart && ctx.chart.width) ? ctx.chart.width : 1000;
                                    return { size: w < 520 ? 10 : 12, weight: '600' };
                                }
                            },
                            grid: { color: 'rgba(15,23,42,.08)' }
                        }
                    }
                }
            });
        }

        // 4. Doughnut Charts
        const makeDoughnut = (ctxId, lbData) => {
            const ctx = document.getElementById(ctxId).getContext('2d');
            return new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: lbData.slice(0, 10).map(d => d.name), // Top 10 for doughnuts
                    datasets: [{ data: lbData.slice(0, 10).map(d => d.count), backgroundColor: colors, borderWidth: 0 }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { position: 'right', labels: { boxWidth: 12, font: {size: 11} } } }
                }
            });
        };

        charts.domains = makeDoughnut('domainsChart', data.leaderboards.domains);
        charts.modalities = makeDoughnut('modalitiesChart', data.leaderboards.modalities);
        // Defensive merge for any lingering schema sentinel
        if (Array.isArray(data.leaderboards.contexts)) {
            const acc = new Map();
            for (const it of data.leaderboards.contexts) {
                const name = String(it.name || '').trim();
                const key = (name.toLowerCase() === 'not_applicable') ? 'Other' : name;
                const prev = acc.get(key) || 0;
                acc.set(key, prev + (Number(it.count) || 0));
            }
            data.leaderboards.contexts = Array.from(acc.entries())
              .map(([name, count]) => ({ name, count }))
              .sort((a, b) => (Number(b.count) || 0) - (Number(a.count) || 0));
        }
        charts.contexts = makeDoughnut('contextsChart', data.leaderboards.contexts);

        // 5. Render Complex Profile Tables
        const y = data.recent_years;
        document.getElementById('authorProfilesContainer').innerHTML = renderComplexProfileTable(data.profiles.authors, y, 'author');
        document.getElementById('institutionProfilesContainer').innerHTML = renderComplexProfileTable(data.profiles.institutions, y, 'institution');

        // 6. Render Bottom Leaderboards
        const speciesLb = Array.isArray(data.leaderboards.species)
          ? data.leaderboards.species.filter(d => String(d.name || '').trim().toLowerCase() !== 'others')
          : [];
        document.getElementById('leaderboardsContainer').innerHTML = `
            ${renderLeaderboard('🐾 Top Species', speciesLb)}
            ${renderLeaderboard('🏷️ Top Features', data.leaderboards.features)}
        `;

    } catch (err) {
        console.error('Failed to load extensive analytics', err);
    }
}

document.addEventListener('DOMContentLoaded', loadAnalytics);
