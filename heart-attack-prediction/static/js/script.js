/* Heart Attack Risk Prediction System: client-side behaviour
 * - theme toggle (dark / light, remembered per browser)
 * - mobile sidebar
 * - form validation feedback and double-submit protection
 * - confirm dialogs for destructive actions
 * - Chart.js charts (theme-aware)
 * Every storage access is wrapped in try/catch: the app works without it.
 */
(function () {
  'use strict';

  var root = document.documentElement;
  var THEME_KEY = 'hr-theme';

  /* ---------------- Theme ---------------- */
  function storedTheme() {
    try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; }
  }
  function saveTheme(value) {
    try { localStorage.setItem(THEME_KEY, value); } catch (e) { /* ignore */ }
  }
  function systemTheme() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function currentTheme() {
    return root.dataset.theme === 'dark' ? 'dark' : 'light';
  }
  function applyTheme(theme) {
    root.dataset.theme = theme;
    root.dataset.bsTheme = theme;
    var icon = document.getElementById('themeIcon');
    if (icon) icon.className = theme === 'dark' ? 'bi bi-sun' : 'bi bi-moon-stars';
    document.dispatchEvent(new CustomEvent('themechange', { detail: { theme: theme } }));
  }

  var initial = storedTheme();
  applyTheme(initial === 'light' || initial === 'dark' ? initial : systemTheme());

  var themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      saveTheme(next);
      applyTheme(next);
    });
  }

  /* ---------------- Sidebar (mobile) ---------------- */
  var sidebar = document.getElementById('sidebar');
  var toggle = document.getElementById('sidebarToggle');
  var backdrop = document.getElementById('sidebarBackdrop');

  function setSidebar(open) {
    if (!sidebar) return;
    sidebar.classList.toggle('open', open);
    if (backdrop) backdrop.hidden = !open;
    if (toggle) toggle.setAttribute('aria-expanded', String(open));
  }
  if (toggle) toggle.addEventListener('click', function () { setSidebar(!sidebar.classList.contains('open')); });
  if (backdrop) backdrop.addEventListener('click', function () { setSidebar(false); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') setSidebar(false); });

  /* ---------------- Tooltips ---------------- */
  if (window.bootstrap) {
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
      new window.bootstrap.Tooltip(el);
    });
  }

  /* ---------------- Forms ---------------- */
  document.querySelectorAll('form.needs-validation').forEach(function (form) {
    form.addEventListener('submit', function (event) {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
        form.classList.add('was-validated');
        var firstInvalid = form.querySelector(':invalid');
        if (firstInvalid) firstInvalid.focus();
        return;
      }
      var btn = document.getElementById('submitBtn');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" aria-hidden="true"></span>Analyzing…';
      }
    });
  });

  /* ---------------- Confirm dialogs (delete actions) ---------------- */
  document.addEventListener('submit', function (e) {
    var message = e.target && e.target.dataset ? e.target.dataset.confirm : null;
    if (message && !window.confirm(message)) e.preventDefault();
  });

  /* ---------------- Charts ---------------- */
  var charts = [];

  function cssVar(name) {
    return getComputedStyle(root).getPropertyValue(name).trim();
  }

  function destroyCharts() {
    charts.forEach(function (c) { c.destroy(); });
    charts = [];
  }

  function buildCharts() {
    destroyCharts();
    if (typeof window.Chart === 'undefined') return;
    var dataEl = document.getElementById('chart-data');
    if (!dataEl) return;

    var data;
    try { data = JSON.parse(dataEl.textContent); } catch (e) { return; }

    var text = cssVar('--muted');
    var grid = cssVar('--chart-grid');
    window.Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
    window.Chart.defaults.color = text;

    var riskCanvas = document.getElementById('riskChart');
    if (riskCanvas && data.positive + data.negative > 0) {
      charts.push(new window.Chart(riskCanvas, {
        type: 'doughnut',
        data: {
          labels: ['Elevated (class 1)', 'Not elevated (class 0)'],
          datasets: [{
            data: [data.positive, data.negative],
            backgroundColor: [cssVar('--danger'), cssVar('--success')],
            borderWidth: 0,
            hoverOffset: 8
          }]
        },
        options: {
          cutout: '68%',
          plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, color: text } } }
        }
      }));
    }

    var probCanvas = document.getElementById('probChart');
    if (probCanvas && data.labels && data.labels.length) {
      charts.push(new window.Chart(probCanvas, {
        type: 'bar',
        data: {
          labels: data.labels,
          datasets: [{
            label: 'Predictions',
            data: data.bins,
            backgroundColor: cssVar('--accent'),
            borderRadius: 8,
            maxBarThickness: 48
          }]
        },
        options: {
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { title: function (items) { return 'Output score ' + items[0].label; } } }
          },
          scales: {
            x: { title: { display: true, text: 'Model output score band' }, grid: { display: false }, ticks: { color: text } },
            y: { beginAtZero: true, ticks: { precision: 0, color: text }, grid: { color: grid },
                 title: { display: true, text: 'Predictions' } }
          }
        }
      }));
    }
  }

  document.addEventListener('themechange', buildCharts);
  buildCharts();
})();
