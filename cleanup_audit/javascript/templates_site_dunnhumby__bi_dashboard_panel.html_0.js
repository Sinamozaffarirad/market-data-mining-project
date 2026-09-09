
const BI = {
  filters: {},                       // the one shared selection
  productLevels: [], timeLevels: [], // level keys, so a crumb knows what to clear
  product: null, time: null, segment: null, store: null, basket: null,
  hour: null, weekday: null, discount: null, brand: null, demo: null,
  pareto: null, growth: null, repeat: null, household: null,
  discountMix: null, brandMix: null,
  page: 'overview',
};
if (window.Chart) {
  Chart.defaults.devicePixelRatio = chartRenderPixelRatio();
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = 12;
  Chart.defaults.color = '#5a6180';
}

const T = {
  indigo: '#667eea', violet: '#8b5cf6', emerald: '#0ea373', amber: '#e08a00',
  rose: '#f43f5e', cyan: '#0891b2',
};
const PALETTE = [T.indigo, T.violet, T.emerald, T.amber, T.rose, T.cyan, '#4a54c9', '#db2777'];

/* ---------- filter context ------------------------------------------------ */

function biParams(extra) {
  const params = new URLSearchParams();
  Object.entries(BI.filters).forEach(([k, v]) => params.set(k, v));
  (extra || []).forEach(([k, v]) => params.append(k, v));
  return params;
}

/* A refresh fires about twenty queries at once and they finish out of order.
   Clicking a second mark before the first refresh has landed used to let the
   older, slower answers overwrite the newer ones, leaving panels showing a
   filter that is no longer selected. Each refresh takes a number, and a reply
   from a superseded one is dropped instead of drawn. */
let biRun = 0;
const BI_STALE = Symbol('superseded refresh');

async function biGet(url, extra) {
  const run = biRun;
  const response = await fetch(`${url}?${biParams(extra).toString()}`);
  const text = await response.text();
  if (run !== biRun) throw BI_STALE;
  try {
    var payload = JSON.parse(text);
  } catch (error) {
    // A server error returns an HTML page; name the endpoint rather than
    // surfacing a raw JSON parse failure that says nothing useful.
    throw new Error(`${url.split('/').filter(Boolean).pop()} failed with HTTP ${response.status}`);
  }
  if (!payload.success) throw new Error(payload.error || 'Request failed');
  return payload;
}

/* Holding ctrl, cmd or shift while clicking a mark adds it to the selection
   instead of replacing it, the way a slicer multi-selects. The modifier is read
   from the last mousedown so every existing chart handler gets it without
   having to thread the event through. */
document.addEventListener('mousedown', event => {
  BI.additive = !!(event.ctrlKey || event.metaKey || event.shiftKey);
}, true);

function biSetFilter(key, value, options) {
  const clean = (value === null || value === undefined) ? '' : String(value).trim();
  const current = BI.filters[key] ? String(BI.filters[key]).split('|') : [];
  if (!clean || clean.toLowerCase() === 'all') {
    delete BI.filters[key];
  } else if (BI.additive && current.length) {
    const next = current.includes(clean)
      ? current.filter(v => v !== clean)     // ctrl-click an active mark to drop it
      : current.concat(clean);
    if (next.length) BI.filters[key] = next.join('|');
    else delete BI.filters[key];
  } else if (current.length === 1 && current[0] === clean && (options || {}).toggle) {
    delete BI.filters[key];
  } else {
    BI.filters[key] = clean;
  }
  biSyncSlicers();
  biRefresh();
}

/* Some marks stand for a combination rather than a single value: a bar on the
   month chart means one month of one year. Setting the parts one at a time
   would refresh the dashboard twice and, in between, show that month across
   every year. This applies the whole set in one go and refreshes once. */
function biSetFilters(pairs, options) {
  const entries = pairs
    .map(([key, value]) => [key, (value === null || value === undefined) ? '' : String(value).trim()])
    .filter(([, value]) => value !== '');
  if (!entries.length) return;
  const already = entries.every(([key, value]) => String(BI.filters[key] || '') === value);
  if (already && (options || {}).toggle) entries.forEach(([key]) => delete BI.filters[key]);
  else entries.forEach(([key, value]) => { BI.filters[key] = value; });
  biSyncSlicers();
  biRefresh();
}

function biRemoveFilter(key) {
  /* Hierarchy levels only make sense with their parents in place, so removing
     one level also drops everything below it. */
  [BI.productLevels, BI.timeLevels].forEach(levels => {
    const index = levels.indexOf(key);
    if (index >= 0) levels.slice(index).forEach(k => delete BI.filters[k]);
  });
  delete BI.filters[key];
  biSyncSlicers();
  biRefresh();
}

function biClearAll() { BI.filters = {}; biSyncSlicers(); biRefresh(); }

/* Keep the dropdowns showing whatever the chips say, so a filter set by
   clicking a chart is reflected in the controls and the other way round. */
function biSyncSlicers() {
  const bound = [
    ['fYear', 'year'], ['fQuarter', 'quarter'], ['fMonth', 'month'],
    ['fDept', 'department'], ['fCommodity', 'commodity'], ['fSegment', 'segment'],
    ['fBrand', 'brand'], ['fWeekday', 'weekday'], ['fHour', 'hour'],
    ['fIncome', 'income'], ['fAge', 'age'], ['fHouseholdSize', 'household_size'],
    ['fStore', 'store'],
  ];
  bound.forEach(([id, key]) => {
    const el = document.getElementById(id);
    if (!el) return;
    const value = BI.filters[key] || '';
    /* A slicer shows a single choice. Several values can be selected by
       ctrl-clicking marks, and the control steps back rather than pretending
       one of them is the selection; the chips carry the full set. */
    const many = value.includes('|');
    if (el.tagName === 'SELECT') {
      el.value = (!many && [...el.options].some(o => o.value === value)) ? value : 'all';
    } else {
      el.value = many ? '' : value;
    }
    el.closest('.field')?.classList.toggle('is-set', !!value);
    if (many) el.title = 'Several selected: ' + value.split('|').join(', ');
    else el.removeAttribute('title');
  });
  biNarrowCommodities();
  const more = document.getElementById('biSlicersMore');
  const secondaryActive = ['brand', 'weekday', 'hour', 'income', 'age', 'household_size', 'store']
    .some(key => BI.filters[key]);
  if (secondaryActive && more?.hidden) biToggleMoreFilters();
}

/* Three hundred commodities is not a list anyone reads. Choosing a department
   narrows it to that department's own, which is also what clicking through the
   product hierarchy does. */
function biNarrowCommodities() {
  const select = document.getElementById('fCommodity');
  if (!select) return;
  const department = BI.filters.department;
  const single = department && !department.includes('|') ? department : null;
  let visible = 0;
  [...select.options].forEach(option => {
    if (!option.dataset.department) return;
    const show = !single || option.dataset.department === single;
    option.hidden = !show;
    if (show) visible += 1;
  });
  const first = select.options[0];
  if (first) {
    first.textContent = single ? `All ${visible} in ${single}` : 'All commodities';
  }
}

function biToggleMoreFilters() {
  const box = document.getElementById('biSlicersMore');
  const button = document.getElementById('biMoreFiltersBtn');
  if (!box) return;
  box.hidden = !box.hidden;
  button?.setAttribute('aria-expanded', String(!box.hidden));
  if (button) {
    button.innerHTML = box.hidden
      ? '<i class="fas fa-sliders"></i> More filters'
      : '<i class="fas fa-sliders"></i> Fewer filters';
  }
}

const FILTER_LABELS = {
  year: 'Year', quarter: 'Quarter', month: 'Month', week: 'Week', day: 'Day',
  weekday: 'Weekday', hour: 'Hour', department: 'Department', commodity: 'Commodity',
  sub_commodity: 'Sub-commodity', product: 'Product', brand: 'Brand', store: 'Store',
  segment: 'Segment', age: 'Age', income: 'Income', household_size: 'Household size',
  period: '30-day period',
};

function renderChips() {
  const box = document.getElementById('biChips');
  const keys = Object.keys(BI.filters);
  box.classList.toggle('empty', !keys.length);
  if (!keys.length) { box.innerHTML = ''; return; }
  box.innerHTML = '<span class="label">Filters</span>' + keys.map(key => `
    <span class="bi-chip"><span class="k">${FILTER_LABELS[key] || key}:</span>
      ${escapeHtml(String(BI.filters[key]).split('|').join(', '))}
      <button type="button" title="Remove" onclick="biRemoveFilter('${key}')">&times;</button>
    </span>`).join('') +
    `<button class="btn btn-sm btn-link text-decoration-none p-0 ms-1" type="button"
       onclick="biClearAll()">Clear all</button>`;
}

/* ---------- formatting ---------------------------------------------------- */

const money = v => '$' + Number(v || 0).toLocaleString(undefined, {maximumFractionDigits: 0});
const money2 = v => '$' + Number(v || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
const pct = v => (Number(v || 0) * 100).toFixed(1) + '%';
const num = v => Number(v || 0).toLocaleString();
function escapeHtml(text) {
  return String(text ?? '').replace(/[&<>"']/g, c => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
  ));
}

/* ---------- pages --------------------------------------------------------- */

function biShowPage(page, button) {
  BI.page = page;
  document.querySelectorAll('.bi-page').forEach(el => el.classList.remove('active'));
  document.getElementById('biPage-' + page)?.classList.add('active');
  document.querySelectorAll('.bi-page-btn').forEach(el => el.classList.remove('active'));
  button?.classList.add('active');
  // Charts on a hidden page could not measure their container, so settle them
  // now the page is on screen.
  setTimeout(() => biSettleCharts(), 60);
  /* Opening the tests page runs the search once, so the reader arrives at
     findings rather than at a button. It is not run before the page is opened,
     because it is the slowest panel here and most visits never reach it. */
  if (page === 'tests' && !biScanRequested) setTimeout(() => loadSignificanceScan(), 80);
}

/* ---------- panels -------------------------------------------------------- */

const KPI_CARDS = [
  ['revenue',               'Revenue',              'fa-sack-dollar',   T.indigo,  money,  k => money(k.revenue_per_day) + '/day'],
  ['baskets',               'Baskets',              'fa-basket-shopping', T.cyan,  num,    () => ''],
  ['avg_basket_value',      'Avg basket value',     'fa-receipt',       T.emerald, money2, () => ''],
  ['avg_basket_size',       'Avg basket size',      'fa-layer-group',   T.emerald, v => Number(v).toFixed(2), () => 'distinct products'],
  ['households',            'Households',           'fa-house-user',    T.violet,  num,    k => money2(k.revenue_per_household) + ' each'],
  ['visits_per_household',  'Visits per household', 'fa-repeat',        T.violet,  v => Number(v).toFixed(1), () => ''],
  ['discount_rate',         'Discount rate',        'fa-percent',       T.rose,    pct,    () => 'of list value'],
  ['top20_concentration',   'Top-20 share',         'fa-ranking-star',  T.amber,   pct,    k => 'of ' + num(k.products) + ' products'],
];

function renderKpis(k) {
  document.getElementById('kpiStrip').innerHTML = KPI_CARDS.map(
    ([key, label, icon, accent, format, hint]) => `
    <div class="kpi" style="--kpi-accent:${accent}">
      <div class="kpi-top">
        <span class="kpi-icon"><i class="fas ${icon}"></i></span>
        <span class="l">${label}</span>
      </div>
      <div class="v">${format(k[key])}</div>
      <div class="h">${hint(k)}</div>
    </div>`).join('');
}

function renderCrumbs(containerId, breadcrumb, rootLabel, onJump) {
  const parts = [`<span class="crumb ${breadcrumb.length ? '' : 'active'}" onclick="${onJump}(0)">${rootLabel}</span>`];
  breadcrumb.forEach((item, index) => {
    parts.push('<span class="crumb-sep">/</span>');
    parts.push(`<span class="crumb ${index === breadcrumb.length - 1 ? 'active' : ''}"
      onclick="${onJump}(${index + 1})">${escapeHtml(item.display || item.value)}</span>`);
  });
  document.getElementById(containerId).innerHTML = parts.join('');
}

function renderDetailTable(rows, levelLabel, levelKey) {
  const body = document.getElementById('detailRows');
  const shownHead = document.getElementById('detailShareShownHead');
  if (shownHead) shownHead.textContent = `Share of these ${rows.length}`;
  document.getElementById('detailSub').textContent = `${levelLabel} breakdown, ranked by revenue`;
  document.getElementById('detailLevelHead').textContent = levelLabel;
  if (!rows.length) { body.innerHTML = '<tr><td colspan="5" class="bi-loading">No rows.</td></tr>'; return; }
  const max = Math.max(...rows.map(r => r.revenue));
  body.innerHTML = rows.map(r => `<tr onclick="biSetFilter('${levelKey}', ${JSON.stringify(String(r.value ?? r.label)).replace(/"/g, '&quot;')}, {toggle:true})">
    <td class="bar-cell"><div class="fill" style="width:${max ? (r.revenue / max * 100) : 0}%"></div><span>${escapeHtml(r.label)}</span></td>
    <td class="n">${money(r.revenue)}</td><td class="n">${pct(r.share)}</td>
    <td class="n">${r.share_of_total === undefined ? '--' : pct(r.share_of_total)}</td>
    <td class="n">${num(r.baskets)}</td>
  </tr>`).join('');
}

async function loadProduct() {
  const data = await biGet('/analysis/api/bi/product-drill/');
  BI.productLevels = data.level_keys;
  renderCrumbs('productCrumbs', data.breadcrumb, 'All departments', 'jumpProduct');
  document.getElementById('productHint').textContent = data.can_drill
    ? `Click a bar to open ${data.next_label}` : 'Deepest level reached';
  renderDetailTable(data.rows, data.level_label, data.level);

  if (BI.product) BI.product.destroy();
  BI.product = new Chart(document.getElementById('productChart'), {
    type: 'bar',
    data: { labels: data.rows.map(r => r.label),
      datasets: [{ label: 'Revenue', data: data.rows.map(r => r.revenue), backgroundColor: T.emerald,
        borderRadius: 4 }] },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      // Clicking a level sets that level's filter, which both descends the
      // hierarchy and narrows every other panel.
      onClick: (event, elements) => {
        if (!elements.length) return;
        biSetFilter(data.level, data.rows[elements[0].index].value, {toggle: true});
      },
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${money(ctx.raw)} · ${pct(data.rows[ctx.dataIndex].share)} of level` } } },
      scales: { x: { beginAtZero: true, ticks: { callback: v => money(v) } } },
    },
  });
}

async function loadTime() {
  const data = await biGet('/analysis/api/bi/time-drill/');
  BI.timeLevels = data.level_keys;
  renderCrumbs('timeCrumbs', data.breadcrumb, 'All years', 'jumpTime');
  document.getElementById('timeHint').textContent = data.can_drill
    ? `Click a column to open ${data.next_label}` : 'Deepest level reached';

  if (BI.time) BI.time.destroy();
  BI.time = new Chart(document.getElementById('timeChart'), {
    type: (data.level === 'day' || data.level === 'week') ? 'line' : 'bar',
    data: { labels: data.rows.map(r => r.label),
      datasets: [{ label: 'Revenue', data: data.rows.map(r => r.revenue),
        backgroundColor: data.rows.map(r => r.partial ? T.amber : T.indigo),
        borderColor: T.indigo, tension: .25, fill: false, borderRadius: 4 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      onClick: (event, elements) => {
        if (!elements.length) return;
        biSetFilter(data.level, data.rows[elements[0].index].value, {toggle: true});
      },
      plugins: { legend: { display: false },
        tooltip: { callbacks: {
          label: ctx => `${money(ctx.raw)} · ${num(data.rows[ctx.dataIndex].baskets)} baskets`,
          afterLabel: ctx => {
            const row = data.rows[ctx.dataIndex];
            const base = `${row.days} day${row.days === 1 ? '' : 's'} · ${money(row.revenue_per_day)}/day`;
            return row.partial ? `${base}\n⚠ shorter period than its neighbours` : base;
          },
        } } },
      scales: { y: { beginAtZero: true, ticks: { callback: v => money(v) } } },
    },
  });
}

/* A crumb clears its own level and everything below it. */
function jumpProduct(depth) { biJump(BI.productLevels, depth); }
function jumpTime(depth) { biJump(BI.timeLevels, depth); }
function biJump(levels, depth) {
  levels.slice(depth).forEach(key => delete BI.filters[key]);
  biSyncSlicers();
  biRefresh();
}

async function loadSegments() {
  const data = await biGet('/analysis/api/bi/segments/');
  if (BI.segment) BI.segment.destroy();
  BI.segment = new Chart(document.getElementById('segmentChart'), {
    type: 'bar',
    data: { labels: data.rows.map(r => r.segment),
      datasets: [
        { label: 'Revenue', data: data.rows.map(r => r.revenue), backgroundColor: T.violet, yAxisID: 'y', order: 2, borderRadius: 4 },
        { label: 'Households', data: data.rows.map(r => r.households), type: 'line',
          borderColor: T.amber, backgroundColor: T.amber, yAxisID: 'y1', order: 1 },
      ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      onClick: (e, els) => { if (els.length) biSetFilter('segment', data.rows[els[0].index].segment, {toggle: true}); },
      plugins: { legend: { position: 'bottom' },
        tooltip: { callbacks: { afterLabel: ctx => `${money2(data.rows[ctx.dataIndex].revenue_per_household)} per household` } } },
      scales: {
        y: { beginAtZero: true, position: 'left', ticks: { callback: v => money(v) } },
        y1: { beginAtZero: true, position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'Households' } },
      },
    },
  });
}

async function loadStores() {
  const data = await biGet('/analysis/api/bi/stores/');
  const maxRevenue = Math.max(...data.rows.map(r => r.revenue), 1);
  if (BI.store) BI.store.destroy();
  BI.store = new Chart(document.getElementById('storeChart'), {
    type: 'bubble',
    data: { datasets: [{ label: 'Store',
      /* The radius used to rise linearly with revenue share, but a circle is
         read by its area, and area grows with the square of the radius. Two
         stores 1.2x apart in revenue were drawn 1.4x apart to the eye. Solving
         for area instead -- area = min + (max - min) * share -- gives
         r = sqrt(rMin^2 + (rMax^2 - rMin^2) * share), which keeps the same 4px
         floor and 20px ceiling while making the area, not the radius, carry
         the number. The floor is still a floor: stores near zero revenue are
         drawn larger than truth so they stay clickable. */
      data: data.rows.map(r => ({ x: r.baskets, y: r.avg_basket,
        r: Math.sqrt(16 + 384 * (r.revenue / maxRevenue)),
        store: r.store_id, revenue: r.revenue })),
      backgroundColor: 'rgba(244,63,94,.42)', borderColor: T.rose }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      onClick: (e, els) => { if (els.length) biSetFilter('store', data.rows[els[0].index].store_id, {toggle: true}); },
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: ctx => {
          const p = ctx.raw;
          return `Store ${p.store} · ${num(p.x)} baskets · ${money2(p.y)} avg · ${money(p.revenue)}`;
        } } } },
      scales: {
        x: { title: { display: true, text: 'Baskets' }, beginAtZero: true },
        y: { title: { display: true, text: 'Average basket value' }, ticks: { callback: v => money2(v) } },
      },
    },
  });
}

async function loadBaskets() {
  const data = await biGet('/analysis/api/bi/basket-distribution/');
  const shown = data.rows.filter(r => r.bucket <= data.cutoff);
  const caption = document.getElementById('basketCaption');
  if (caption) {
    caption.innerHTML = data.tail.baskets
      ? `Median basket holds <strong>${data.median_size}</strong> products. The largest holds <strong>${data.max_size}</strong>.
         Sizes above ${data.cutoff} are left off the bars &mdash; <strong>${num(data.tail.baskets)}</strong>
         baskets (${pct(data.tail.share)}), ${money(data.tail.revenue)}.`
      : `Median basket holds <strong>${data.median_size}</strong> products. The largest holds <strong>${data.max_size}</strong>.`;
  }
  if (BI.basket) BI.basket.destroy();
  BI.basket = new Chart(document.getElementById('basketChart'), {
    type: 'bar',
    data: { labels: shown.map(r => r.bucket),
      datasets: [{ label: 'Baskets', data: shown.map(r => r.baskets), backgroundColor: T.cyan, borderRadius: 3 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { callbacks: {
          title: items => `${items[0].label} distinct products`,
          label: ctx => `${num(ctx.raw)} baskets · ${money(shown[ctx.dataIndex].revenue)}`,
        } } },
      scales: {
        x: { title: { display: true, text: 'Distinct products in basket' } },
        y: { beginAtZero: true, title: { display: true, text: 'Baskets' } },
      },
    },
  });
}

async function loadDaypart() {
  const data = await biGet('/analysis/api/bi/daypart/');
  if (BI.hour) BI.hour.destroy();
  BI.hour = new Chart(document.getElementById('hourChart'), {
    type: 'bar',
    data: { labels: data.hours.map(r => String(r.hour).padStart(2, '0') + ':00'),
      datasets: [{ label: 'Revenue', data: data.hours.map(r => r.revenue), backgroundColor: T.amber, borderRadius: 4 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      onClick: (e, els) => { if (els.length) biSetFilter('hour', data.hours[els[0].index].hour, {toggle: true}); },
      plugins: { legend: { display: false },
        tooltip: { callbacks: { afterLabel: ctx => `${num(data.hours[ctx.dataIndex].baskets)} baskets · ${money2(data.hours[ctx.dataIndex].avg_basket)} avg` } } },
      scales: { y: { beginAtZero: true, ticks: { callback: v => money(v) } } },
    },
  });

  if (BI.weekday) BI.weekday.destroy();
  BI.weekday = new Chart(document.getElementById('weekdayChart'), {
    type: 'bar',
    // Weekdays recur a different number of times across the window, so the
    // comparable measure is revenue per occurrence, not the raw total.
    data: { labels: data.weekdays.map(r => r.weekday),
      datasets: [{ label: 'Revenue per day', data: data.weekdays.map(r => r.revenue_per_day),
        backgroundColor: T.emerald, borderRadius: 4 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      onClick: (e, els) => { if (els.length) biSetFilter('weekday', data.weekdays[els[0].index].weekday, {toggle: true}); },
      plugins: { legend: { display: false },
        tooltip: { callbacks: { afterLabel: ctx => `${data.weekdays[ctx.dataIndex].days} occurrences · ${money(data.weekdays[ctx.dataIndex].revenue)} total` } } },
      scales: { y: { beginAtZero: true, ticks: { callback: v => money(v) } } },
    },
  });
}

async function loadDiscountTrend() {
  const data = await biGet('/analysis/api/bi/discount-trend/');
  if (BI.discount) BI.discount.destroy();
  BI.discount = new Chart(document.getElementById('discountChart'), {
    data: { labels: data.rows.map(r => r.label),
      datasets: [
        { type: 'bar', label: 'Revenue', data: data.rows.map(r => r.revenue),
          backgroundColor: data.rows.map(r => r.partial ? T.amber : T.indigo), yAxisID: 'y', order: 2, borderRadius: 4 },
        { type: 'line', label: 'Discount rate', data: data.rows.map(r => r.discount_rate),
          borderColor: T.rose, backgroundColor: T.rose, yAxisID: 'y1', tension: .25, order: 1 },
      ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      /* A bar is one month of one year, so it sets both halves of the time
         hierarchy; the chips then read the same way as drilling down to it. */
      onClick: (e, els) => {
        if (!els.length) return;
        const row = data.rows[els[0].index];
        biSetFilters([['year', row.calendar_year], ['month', row.month_name]], {toggle: true});
      },
      plugins: { legend: { position: 'bottom' },
        tooltip: { callbacks: { afterLabel: ctx => {
          const row = data.rows[ctx.dataIndex];
          return row.partial ? `${row.days} days · shorter month in the window` : `${row.days} days`;
        } } } },
      scales: {
        y: { beginAtZero: true, position: 'left', ticks: { callback: v => money(v) } },
        y1: { beginAtZero: true, position: 'right', grid: { drawOnChartArea: false },
              ticks: { callback: v => pct(v) }, title: { display: true, text: 'Discount rate' } },
      },
    },
  });
}

async function loadBrand() {
  const data = await biGet('/analysis/api/bi/brand/');
  const note = document.getElementById('brandNote');
  if (note) {
    note.innerHTML = data.rows.map(r =>
      `<strong>${escapeHtml(r.label)}</strong> ${pct(r.share)} of revenue, ${pct(r.discount_rate)} discounted`
    ).join(' &middot; ');
  }
  if (BI.brand) BI.brand.destroy();
  BI.brand = new Chart(document.getElementById('brandChart'), {
    type: 'doughnut',
    data: { labels: data.rows.map(r => r.label),
      datasets: [{ data: data.rows.map(r => r.revenue), backgroundColor: [T.emerald, T.indigo, T.amber, T.rose] }] },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '58%',
      onClick: (e, els) => { if (els.length) biSetFilter('brand', data.rows[els[0].index].label, {toggle: true}); },
      plugins: { legend: { position: 'bottom' },
        tooltip: { callbacks: { label: ctx => `${ctx.label}: ${money(ctx.raw)} · ${pct(data.rows[ctx.dataIndex].share)}` } } },
    },
  });
}

async function loadDemographics() {
  const dimension = document.getElementById('demoDim')?.value || 'age';
  const data = await biGet('/analysis/api/bi/demographics/', [['dimension', dimension]]);
  document.getElementById('demoSub').textContent = `Revenue by ${data.dimension_label.toLowerCase()}`;
  const note = document.getElementById('demoNote');
  if (note) {
    // Only 802 of 2,497 households carry demographics, so state the covered
    // share rather than letting the reader assume the chart is everyone.
    note.textContent = `Covers ${pct(data.coverage)} of revenue in this selection; `
      + 'the rest belongs to households with no demographic record.';
  }
  if (BI.demo) BI.demo.destroy();
  BI.demo = new Chart(document.getElementById('demoChart'), {
    type: 'bar',
    data: { labels: data.rows.map(r => r.label),
      datasets: [{ label: 'Revenue', data: data.rows.map(r => r.revenue), borderRadius: 4,
        backgroundColor: data.rows.map((_, i) => PALETTE[i % PALETTE.length]) }] },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      onClick: (e, els) => { if (els.length) biSetFilter(data.dimension, data.rows[els[0].index].label, {toggle: true}); },
      plugins: { legend: { display: false },
        tooltip: { callbacks: { afterLabel: ctx => `${num(data.rows[ctx.dataIndex].households)} households · ${money2(data.rows[ctx.dataIndex].revenue_per_household)} each` } } },
      scales: { x: { beginAtZero: true, ticks: { callback: v => money(v) } } },
    },
  });
}

async function loadTopProducts() {
  const data = await biGet('/analysis/api/bi/top-products/');
  const body = document.getElementById('topProductRows');
  /* The count is read back rather than written into the header: a filter can
     leave fewer than 25 rows, and a header that still said 25 would be naming
     the wrong base for the column beneath it. */
  const shownHead = document.getElementById('topShareShownHead');
  if (shownHead) shownHead.textContent = `Share of these ${Number(data.shown || data.rows.length)}`;
  body.innerHTML = data.rows.length ? data.rows.map(r => `<tr onclick="biSetFilter('product', '${r.product_id}', {toggle:true})">
    <td>${r.rank}</td><td><strong>${r.product_id}</strong></td>
    <td>${escapeHtml(r.department)}</td><td>${escapeHtml(r.commodity)}</td><td>${escapeHtml(r.brand)}</td>
    <td class="n">${money(r.revenue)}</td><td class="n">${pct(r.share)}</td>
    <td class="n">${r.share_of_total === undefined ? '--' : pct(r.share_of_total)}</td>
    <td class="n">${num(r.baskets)}</td>
  </tr>`).join('') : '<tr><td colspan="9" class="bi-loading">No products match these filters.</td></tr>';
}

async function loadInsights() {
  const data = await biGet('/analysis/api/bi/insights/');
  const container = document.getElementById('insightList');
  container.innerHTML = data.insights.length
    ? data.insights.map(i => `<div class="insight"><div class="t">${escapeHtml(i.title)}</div><div class="d">${escapeHtml(i.detail)}</div></div>`).join('')
    : '<div class="bi-loading">No findings for this selection.</div>';
}

/* Significance tests. `reset` clears the chosen groups so switching dimension
   picks that dimension's own two largest groups rather than carrying stale
   names across. */
/* Searches the comparisons rather than waiting to be pointed at one. Ordered
   by effect size: at this many baskets a p-value is significant almost
   everywhere, so sorting by it would rank noise alongside the real gaps. */
/* Once the scan has been run it follows the dashboard: a cross-filter or a
   drill changes which groups are being compared, and a list left as it was
   would be describing a selection the reader has already moved off. It stays
   idle until first asked, because it is the slowest panel here and nobody
   should pay for it before they want it. */
let biScanRequested = false;
/* Filters can change faster than a scan finishes. Each run carries a serial and
   only the newest is allowed to render, otherwise a slow earlier scan lands on
   top of a newer one and describes a selection that is no longer set. */
let biScanSerial = 0;
let biScanShowAll = false;
let biScanLastPayload = null;

/* Re-lays the table already in hand rather than asking the server again. */
function biScanToggleAll(showAll) {
  biScanShowAll = !!showAll;
  if (biScanLastPayload) biScanRender(biScanLastPayload);
}

/* Both panels have to move together. Leaving the scan on basket value while
   the tests below switched to visits would put two answers to different
   questions on one screen, and the scan's rows link straight into those tests.
   The scan is only re-run if it has already been run once, because it is the
   expensive half and is behind a button for that reason. */
function biMeasureChanged() {
  loadSignificance();
  if (biScanRequested) loadSignificanceScan();
}

async function loadSignificanceScan() {
  biScanRequested = true;
  const serial = ++biScanSerial;
  const box = document.getElementById('sigScanResults');
  const summary = document.getElementById('sigScanSummary');
  const button = document.getElementById('sigScanBtn');
  if (!box) return;
  box.innerHTML = '<div class="bi-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Checking every pair&hellip;</div>';
  if (button) { button.disabled = true; }
  try {
    const data = await biGet('/analysis/api/bi/significance-scan/',
      [['measure', document.getElementById('sigMeasure')?.value || 'basket_value']]);
    if (serial !== biScanSerial) return;
    biScanLastPayload = data;
    biScanRender(data);
  } catch (error) {
    if (error !== BI_STALE && serial === biScanSerial) {
      box.innerHTML = `<div class="alert alert-danger py-2 px-3 small mb-0">Scan failed: ${escapeHtml(error.message)}</div>`;
    }
  } finally {
    if (button && serial === biScanSerial) { button.disabled = false; }
  }
}

/* One test per dimension rather than per pair, which answers a question the
   pairwise list cannot: is this way of grouping customers worth slicing by at
   all? */
function renderDimensionRanking(dims) {
  const box = document.getElementById('sigDimRank');
  if (!box) return;
  if (!dims.length) { box.innerHTML = ''; return; }
  box.innerHTML = `
    <h6 class="mb-1">Which way of grouping customers separates spend most</h6>
    <span class="bi-stamp d-block mb-2">One test across all the groups of a dimension at once,
    not a pair at a time.</span>
    <div class="table-responsive"><table class="sig-scan">
      <thead><tr><th>Grouping</th><th class="text-end">Groups</th>
        <th>Share of the spread it explains</th><th class="text-end">Kruskal-Wallis H</th>
        <th class="text-end">ANOVA F</th></tr></thead>
      <tbody>${dims.map(d => `<tr onclick="biJumpToDimension('${escapeHtml(d.dimension)}')"
            title="Compare this dimension's groups below">
        <td><strong>${escapeHtml(d.dimension_label)}</strong></td>
        <td class="text-end">${d.groups}</td>
        <td><div class="gap"><span class="sig-bar"><i style="width:${
            Math.min(100, Number(d.epsilon_squared) / 0.14 * 100).toFixed(1)}%"></i></span>
            <span class="sig-tag ${d.effect}">${(Number(d.epsilon_squared) * 100).toFixed(1)}%</span></div></td>
        <td class="text-end">${Number(d.kruskal_h).toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
        <td class="text-end">${Number(d.anova_f).toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
      </tr>`).join('')}</tbody>
    </table></div>
    <p class="bi-foot">The percentage is how much of the variation in basket value that grouping
    accounts for; the rest is everything else. ANOVA answers the same question assuming a normal
    spread, which basket value does not have, so it sits alongside rather than being relied on.</p>`;
}

/* Points at the test the click actually opened. The pair list ranks on the rank
   test, and the dimension list on Kruskal-Wallis, so those are the cards to
   bring forward -- highlighting the whole panel said only "something changed"
   and highlighting the source row left the reader looking in the wrong place. */
function biHighlightTest(testName) {
  const cards = [...document.querySelectorAll('#sigResults .test-card')];
  if (!cards.length) return;
  cards.forEach(card => card.classList.remove('test-card-picked'));
  const card = cards.find(c => (c.dataset.testName || '').toLowerCase() === testName.toLowerCase())
            || cards[0];
  card.scrollIntoView({behavior: 'smooth', block: 'center'});
  // Wait for the scroll to arrive, or the pop is over before the reader is.
  const onScreen = () => {
    const box = card.getBoundingClientRect();
    return box.top < window.innerHeight * 0.9 && box.bottom > window.innerHeight * 0.1;
  };
  let last = null, still = 0;
  const start = () => {
    void card.offsetWidth;
    card.classList.add('test-card-picked');
  };
  const poll = setInterval(() => {
    const top = Math.round(card.getBoundingClientRect().top);
    still = (top === last) ? still + 1 : 0;
    last = top;
    if (still >= 2 && onScreen()) { clearInterval(poll); start(); }
  }, 90);
  setTimeout(() => { clearInterval(poll);
    if (!card.classList.contains('test-card-picked')) start(); }, 1400);
}

function biJumpToDimension(dimension) {
  const dim = document.getElementById('sigDim');
  if (dim) dim.value = dimension;
  loadSignificance(true).then(() => biHighlightTest('Kruskal-Wallis'));
}

function biScanRender(data) {
  const box = document.getElementById('sigScanResults');
  const summary = document.getElementById('sigScanSummary');
  if (!box) return;
  {
    const rows = data.rows || [];
    if (summary) {
      const circular = (data.rows || []).find(r => r.circular);
      summary.innerHTML = `<strong>${escapeHtml(data.headline)}</strong>
        <span class="d-block mt-1">${escapeHtml(data.method)}</span>`
        + (circular ? `<span class="d-block mt-1 text-warning-emphasis">
            <i class="fas fa-circle-exclamation"></i> ${escapeHtml(circular.circular_note)}</span>` : '');
    }
    if (!rows.length) {
      box.innerHTML = `<div class="bi-loading">No pair of groups has enough
        ${escapeHtml(data.observation_noun || 'baskets')} in this selection to compare.</div>`;
      return;
    }
    /* Medians here are trips when the tab is measuring visits, so the money
       formatter cannot be assumed, and the column headings have to say which
       response the scan was run on. */
    const isCount = data.measure_unit === 'count';
    const scanAmount = value => isCount
      ? Number(value).toLocaleString(undefined, {maximumFractionDigits: 1})
      : money2(value);
    const measureName = data.measure_label || 'Basket value';
    const unitPlural = data.observation_noun || 'baskets';
    const top = Math.max(...rows.map(r => Math.abs(Number(r.delta)))) || 1;
    /* The headline counts the comparisons worth a look, so those are the ones
       laid out. The smaller ones stay one click away rather than padding the
       table or being dropped silently. */
    const worthALook = rows.filter(r => r.actionable);
    const shown = biScanShowAll || !worthALook.length ? rows : worthALook;
    const hidden = rows.length - shown.length;
    box.innerHTML = `<div class="table-responsive"><table class="sig-scan">
      <thead><tr>
        <th>#</th><th>Comparison</th><th>Which is bigger</th>
        <th>Difference in ${escapeHtml(measureName.toLowerCase())}</th>
        <th>Typical ${escapeHtml(unitPlural)}</th>
        <th>What they buy</th><th>Chance of a fluke</th>
      </tr></thead><tbody>${shown.map((row, index) => {
        const delta = Math.abs(Number(row.delta));
        const leaderMedian = row.leader === row.group_a ? row.median_a : row.median_b;
        const trailerMedian = row.leader === row.group_a ? row.median_b : row.median_a;
        return `<tr class="${row.actionable ? '' : 'is-quiet'}"
                 onclick="biOpenComparison('${escapeHtml(row.dimension)}','${escapeHtml(row.group_a).replace(/'/g, "\\'")}','${escapeHtml(row.group_b).replace(/'/g, "\\'")}')"
                 title="Open this comparison in the test below">
          <td>${index + 1}</td>
          <td><strong>${escapeHtml(row.dimension_label)}</strong>
              <small class="d-block text-muted">${escapeHtml(row.group_a)} vs ${escapeHtml(row.group_b)}</small></td>
          <td><strong>${escapeHtml(row.leader)}</strong>
              <small class="d-block text-muted">over ${escapeHtml(row.trailer)}</small>
              ${row.circular ? `<small class="d-block text-muted" title="${escapeHtml(row.circular_note)}"><i class="fas fa-circle-exclamation"></i> partly by definition</small>` : ''}</td>
          <td><div class="gap"><span class="sig-bar"><i style="width:${(delta / top * 100).toFixed(1)}%"></i></span>
              <span class="sig-tag ${row.effect}">${row.effect}</span></div></td>
          <td>${scanAmount(leaderMedian)} <span class="text-muted">vs</span> ${scanAmount(trailerMedian)}</td>
          <td>${row.mix_v === null
              ? '<span class="text-muted" title="Comparing departments by department is circular">--</span>'
              : `<span class="sig-tag ${row.mix_effect}"
                       title="How differently the two groups spread their baskets across departments">${
                   biMixWords(row.mix_effect)}</span>`}</td>
          <td>${row.actionable ? '' : '<span class="text-muted">'}${biFormatQ(row.q_value)}${row.actionable ? '' : '</span>'}</td>
        </tr>`;
      }).join('')}</tbody></table></div>
      <p class="bi-foot">Measured on <strong>${escapeHtml(measureName.toLowerCase())}</strong>,
      one row per ${escapeHtml(unitPlural.replace(/s$/, ''))}; the tests below use the same choice.
      <strong>What they buy</strong> is which departments they shop, which is asked the same way
      whichever measure is chosen. A group can differ sharply on one and not the other, so the
      two columns often disagree.
      Biggest gaps first. Click any row to test that pair in full below.
      ${hidden > 0
        ? `Another <strong>${hidden}</strong> comparison${hidden === 1 ? ' is' : 's are'} real but
           too small to matter.
           <button type="button" class="btn btn-link btn-sm p-0 align-baseline"
                   onclick="biScanToggleAll(true)">Show ${hidden === 1 ? 'it' : 'them'}</button>`
        : (biScanShowAll && worthALook.length
            ? `Showing all ${rows.length}, including the ones too small to matter.
               <button type="button" class="btn btn-link btn-sm p-0 align-baseline"
                       onclick="biScanToggleAll(false)">Show only the ${worthALook.length} worth a look</button>`
            : 'Faded rows are real differences that are too small to matter.')}</p>`;

    renderDimensionRanking(data.dimensions || []);
  }
}

/* The mix column answers a different question from the spend column, and
   reusing "negligible" for both made the pair look contradictory: a group can
   spend far more per basket while filling it from the very same aisles. */
function biMixWords(effect) {
  return {negligible: 'same aisles', small: 'slightly different',
          medium: 'different', large: 'very different'}[effect] || effect;
}

/* A number nobody reads as 4.2e-136. What matters is whether it is a fluke. */
function biFormatQ(value) {
  const q = Number(value);
  if (!isFinite(q)) return '--';
  if (q < 0.0001) return 'under 1 in 10,000';
  if (q < 0.01) return 'under 1 in 100';
  if (q < 0.05) return `about ${(q * 100).toFixed(1)}%`;
  return `${(q * 100).toFixed(0)}% -- could be chance`;
}

/* Hands a scanned pair to the manual panel, so the scan is a starting point
   rather than a separate answer. */
function biOpenComparison(dimension, groupA, groupB) {
  const dim = document.getElementById('sigDim');
  if (dim) dim.value = dimension;
  loadSignificance(true).then(() => {
    const a = document.getElementById('sigA');
    const b = document.getElementById('sigB');
    if (a && [...a.options].some(o => o.value === groupA)) a.value = groupA;
    if (b && [...b.options].some(o => o.value === groupB)) b.value = groupB;
    return loadSignificance();
  }).then(() => biHighlightTest('Mann-Whitney U'));
}

async function loadSignificance(reset) {
  const dimension = document.getElementById('sigDim')?.value || 'segment';
  const measure = document.getElementById('sigMeasure')?.value || 'basket_value';
  const extra = [['dimension', dimension], ['measure', measure]];
  if (!reset) {
    const a = document.getElementById('sigA')?.value;
    const b = document.getElementById('sigB')?.value;
    if (a) extra.push(['group_a', a]);
    if (b) extra.push(['group_b', b]);
  }
  const results = document.getElementById('sigResults');
  results.innerHTML = '<div class="bi-loading">Running tests&hellip;</div>';
  const data = await biGet('/analysis/api/bi/significance/', extra);

  [['sigA', data.group_a], ['sigB', data.group_b]].forEach(([id, selected]) => {
    const select = document.getElementById(id);
    if (!select) return;
    select.innerHTML = data.options.map(o =>
      `<option value="${escapeHtml(o)}" ${o === selected ? 'selected' : ''}>${escapeHtml(o)}</option>`).join('');
  });

  document.getElementById('sigCaveat').innerHTML =
    `<i class="fas fa-triangle-exclamation"></i> ${escapeHtml(data.caveat)}` +
    (data.sampled ? ` Tests use a reproducible sample of up to ${num(data.sample_size)}
       ${escapeHtml(data.observation_noun || 'baskets')} per group. The medians and counts
       quoted are from all of them.` : '')
    + (data.circular ? `<span class="d-block mt-1">
       <i class="fas fa-circle-exclamation"></i> ${escapeHtml(data.circular_note)}</span>` : '');

  /* The card answers the question. The numbers behind it stay folded away, so a
     reader who wants the finding is not made to read the statistics first. */
  results.innerHTML = data.tests.length ? data.tests.map(t => `
    <div class="test-card" data-test-name="${escapeHtml(t.name)}">
      <div class="verdict-row">
        <span class="verdict-badge verdict-${t.verdict}">
          <i class="fas ${t.verdict === 'acted-on' ? 'fa-circle-check' : 'fa-circle-minus'}"></i>
          ${t.verdict === 'acted-on' ? 'Worth acting on' : 'Too small to act on'}
        </span>
        <span class="effect-pill effect-${t.effect_label}">${t.effect_label} difference</span>
      </div>
      <div class="headline">${escapeHtml(t.headline)}</div>
      <div class="q">${escapeHtml(t.question)}</div>
      <details class="test-detail">
        <summary>Show the numbers</summary>
        <div class="test-metrics">
          <div class="m"><div class="mv">${escapeHtml(t.name)}</div><div class="ml">test used</div></div>
          <div class="m"><div class="mv">${t.p_value < 0.001 ? '&lt; 0.001' : t.p_value.toFixed(4)}</div><div class="ml">p-value</div></div>
          ${t.q_value === undefined ? '' : `<div class="m" title="Benjamini-Hochberg across the ${t.comparisons} tests run together here. Three of them read the identical two samples of basket value and two more read those values again across every group, so the raw p-values are not independent and each one alone overstates the evidence.">
            <div class="mv">${t.q_value < 0.001 ? '&lt; 0.001' : Number(t.q_value).toFixed(4)}</div>
            <div class="ml">q-value (of ${t.comparisons})</div></div>`}
          <div class="m"><div class="mv">${Number(t.effect).toFixed(4)}</div><div class="ml">${escapeHtml(t.effect_name)}</div></div>
          <div class="m"><div class="mv">${Number(t.statistic).toLocaleString(undefined,{maximumFractionDigits:3})}</div><div class="ml">statistic</div></div>
        </div>
        <div class="q mb-2">${escapeHtml(t.detail)}</div>
        <div class="q mb-2">${escapeHtml(t.plain)}</div>
        <div class="test-why">${escapeHtml(t.why)}</div>
      </details>
    </div>`).join('')
    : '<div class="bi-loading">Not enough baskets in these groups to test.</div>';

  renderSignificanceTable(data);
}

/* One row per test in the shape a report expects: the two groups, the test, its
   statistic, the p-value, the effect size and the verdict. The cards above are
   for reading; this is the table to quote, and to hand to another tool. */
let lastSignificanceTable = null;

function renderSignificanceTable(data) {
  const box = document.getElementById('sigTable');
  if (!box) return;
  const tests = data.tests || [];
  if (!tests.length) { box.innerHTML = ''; lastSignificanceTable = null; return; }
  const pairFor = t => (/ANOVA|Kruskal/i.test(t.name)
    ? `all ${escapeHtml(data.dimension_label.toLowerCase())} groups`
    : `${escapeHtml(data.group_a)} vs ${escapeHtml(data.group_b)}`);
  /* A confidence interval on visits is a count of trips, not a sum of money,
     so the formatter follows the response the tests were run on. */
  const sigAmount = value => data.measure_unit === 'count'
    ? Number(value).toLocaleString(undefined, {maximumFractionDigits: 1})
    : money2(value);
  lastSignificanceTable = tests.map(t => ({
    group_a: /ANOVA|Kruskal/i.test(t.name) ? 'all groups' : data.group_a,
    group_b: /ANOVA|Kruskal/i.test(t.name) ? 'all groups' : data.group_b,
    dimension: data.dimension_label,
    response_variable: data.measure_label,
    test: t.name,
    statistic: t.statistic,
    p_value: t.p_value,
    q_value: t.q_value,
    comparisons_corrected_across: t.comparisons,
    effect_name: t.effect_name,
    effect_size: t.effect,
    effect_label: t.effect_label,
    confidence_interval: t.confidence_interval
      ? `${t.confidence_interval[0].toFixed(2)} to ${t.confidence_interval[1].toFixed(2)}`
      : '',
    significant: (t.q_value === undefined ? t.p_value : t.q_value) < 0.05 ? 'yes' : 'no',
    worth_acting_on: t.verdict === 'acted-on' ? 'yes' : 'no',
  }));
  box.innerHTML = `
    <details class="rev-advanced sig-table-fold">
    <summary><i class="fas fa-table-list"></i> All ${tests.length} tests side by side</summary>
    <div class="d-flex justify-content-between align-items-end flex-wrap gap-2 mb-2 mt-2">
      <span class="bi-stamp">One row per test, with its statistic, p-value and effect size.
      Response variable: <strong>${escapeHtml(data.measure_label || 'Basket value')}</strong>,
      one row per ${escapeHtml((data.observation_noun || 'baskets').replace(/s$/, ''))}.</span>
      <button class="btn btn-sm btn-outline-secondary" type="button" onclick="exportSignificanceTable()">
        <i class="fas fa-download"></i> Export CSV
      </button>
    </div>
    <div class="table-responsive"><table class="sig-scan">
      <thead><tr>
        <th>Groups compared</th><th>Test</th><th class="text-end">Statistic</th>
        <th class="text-end">P-value</th><th class="text-end">Q-value</th>
        <th>Effect size</th><th class="text-end">95% CI</th>
        <th>Significant</th><th>Worth acting on</th>
      </tr></thead>
      <tbody>${tests.map(t => `<tr>
        <td>${pairFor(t)}</td>
        <td>${escapeHtml(t.name)}</td>
        <td class="text-end">${Number(t.statistic).toLocaleString(undefined, {maximumFractionDigits: 3})}</td>
        <td class="text-end">${t.p_value < 0.001 ? '&lt; 0.001' : Number(t.p_value).toFixed(4)}</td>
        <td class="text-end">${t.q_value === undefined ? '--'
            : (t.q_value < 0.001 ? '&lt; 0.001' : Number(t.q_value).toFixed(4))}</td>
        <td>${escapeHtml(t.effect_name)} ${Number(t.effect).toFixed(4)}
            <span class="sig-tag ${t.effect_label}">${t.effect_label}</span></td>
        <td class="text-end">${t.confidence_interval
            ? sigAmount(t.confidence_interval[0]) + ' to ' + sigAmount(t.confidence_interval[1]) : '--'}</td>
        <td>${(t.q_value === undefined ? t.p_value : t.q_value) < 0.05 ? 'yes' : 'no'}</td>
        <td>${t.verdict === 'acted-on' ? 'yes' : 'no'}</td>
      </tr>`).join('')}</tbody>
    </table></div>
    <p class="bi-foot"><strong>Q-value</strong> is the p-value corrected for running these tests
    together: three of them read the identical two samples of basket value and two more read those
    values again across every group, so each raw p-value on its own counts the same evidence. <strong>Significant</strong> reads the q-value, not the p.
    <strong>Worth acting on</strong> means the difference is also big enough to notice. With this
    many baskets almost everything is significant either way, so the last column is the one to
    read.</p></details>`;
}

function exportSignificanceTable() {
  if (!lastSignificanceTable || !lastSignificanceTable.length) return;
  const columns = Object.keys(lastSignificanceTable[0]);
  const quote = v => {
    const text = String(v ?? '');
    return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  };
  const csv = [columns.join(','),
    ...lastSignificanceTable.map(row => columns.map(c => quote(row[c])).join(','))].join('\n');
  const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'significance_results.csv';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

/* ---------- concentration, growth, loyalty and promotion panels ----------- */

const biDestroy = (chart) => { if (chart) chart.destroy(); };

/* Revenue concentration. The curve is informational rather than a filter: a
   point on it is a rank, not a product anyone would want to select. */
async function loadPareto() {
  const canvas = document.getElementById('paretoChart');
  if (!canvas) return;
  const data = await biGet('/analysis/api/bi/pareto/');
  const note = document.getElementById('paretoMilestones');
  const m = data.milestones || {};
  if (note) {
    note.innerHTML = m.products_for_80pct
      ? `<strong>${num(m.count_for_80pct)}</strong> products (${m.products_for_80pct}% of
         ${num(data.products)}) earn 80% of revenue &middot; half comes from
         <strong>${num(m.count_for_50pct)}</strong>`
      : '';
  }
  biDestroy(BI.pareto);
  BI.pareto = new Chart(canvas, {
    type: 'line',
    data: {
      datasets: [{
        label: 'Cumulative revenue share',
        data: (data.points || []).map(pt => ({x: pt.product_share, y: pt.revenue_share})),
        borderColor: T.violet, backgroundColor: 'rgba(139,92,246,.14)',
        fill: true, pointRadius: 0, borderWidth: 2, tension: .25,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: {type: 'linear', min: 0, max: 100,
            title: {display: true, text: 'Share of products (%)'},
            ticks: {callback: v => v + '%'}},
        y: {min: 0, max: 100, title: {display: true, text: 'Share of revenue (%)'},
            ticks: {callback: v => v + '%'}},
      },
      plugins: {
        legend: {display: false},
        tooltip: {callbacks: {label: c =>
          `Top ${c.parsed.x.toFixed(1)}% of products = ${c.parsed.y.toFixed(1)}% of revenue`}},
      },
    },
  });
}

/* Growth follows the time filter down rather than holding a fixed pair of
   periods: it compares the two newest whole 30-day periods the filter still
   contains, or, when the filter cuts across periods and leaves none whole, the
   newer half of the selected days against the older half. Only a filter too
   narrow to hold two windows comes back empty. The server says which comparison
   it drew, so print that rather than guessing at a reason. */
async function loadGrowth() {
  const canvas = document.getElementById('growthChart');
  if (!canvas) return;
  const dimension = document.getElementById('growthDim')?.value || 'department';
  const sort = document.getElementById('growthSort')?.value || 'absolute';
  const floor = document.getElementById('growthFloor')?.value || '0';
  const data = await biGet('/analysis/api/bi/growth/',
    [['dimension', dimension], ['sort', sort], ['min_revenue', floor]]);
  const sub = document.getElementById('growthSub');
  const rows = (data.rows || []).filter(r => Number(r.current_revenue) || Number(r.previous_revenue));
  if (sub) {
    const left = Number(data.excluded || 0);
    sub.textContent = (data.note || '')
      + (left ? ` · ${left} group${left === 1 ? '' : 's'} below the minimum left out` : '');
  }
  biDestroy(BI.growth);
  if (!rows.length) { BI.growth = null; return; }
  const top = rows.slice(0, 8).concat(rows.slice(-8)).filter(
    (row, index, all) => all.indexOf(row) === index);

  /* The bars have to carry whatever the ranking used. Plotting money while
     ranking by percentage would put the longest bar in the middle of the
     chart and read as a sorting fault. A line with no earlier revenue has no
     finite percentage, so it is drawn at the tallest percentage present
     rather than as a gap; the tooltip still says it had none. */
  const byPercent = (data.sort || sort) === 'percent';
  const finitePct = top.map(r => r.change_pct).filter(v => v !== null).map(Number);
  const pctCeiling = finitePct.length ? Math.max(...finitePct.map(Math.abs)) : 100;
  const measure = row => {
    if (!byPercent) return Number(row.change);
    if (row.change_pct !== null) return Number(row.change_pct);
    return Number(row.current_revenue) > 0 ? pctCeiling : 0;
  };
  BI.growth = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: top.map(r => r.label),
      datasets: [{
        label: byPercent ? 'Percent change' : 'Revenue change',
        data: top.map(measure),
        backgroundColor: top.map(r => Number(r.change) >= 0 ? T.emerald : T.rose),
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      onClick: (event, elements) => {
        if (!elements.length) return;
        const row = top[elements[0].index];
        const key = dimension === 'segment' ? 'segment' : dimension;
        biSetFilter(key, row.label, {toggle: true});
      },
      scales: {x: {
        ticks: {callback: v => byPercent ? `${Number(v).toFixed(0)}%` : money(v)},
        title: {display: true, text: byPercent ? 'Change in revenue (%)' : 'Change in revenue'},
      }},
      plugins: {
        legend: {display: false},
        tooltip: {callbacks: {label: c => {
          const row = top[c.dataIndex];
          const pctText = row.change_pct === null ? 'no earlier revenue'
            : `${Number(row.change_pct) >= 0 ? '+' : ''}${Number(row.change_pct).toFixed(1)}%`;
          return [`Change ${money(row.change)} (${pctText})`,
                  `Now ${money(row.current_revenue)} · before ${money(row.previous_revenue)}`];
        }}},
      },
    },
  });
}

async function loadRepeat() {
  const canvas = document.getElementById('repeatChart');
  if (!canvas) return;
  const data = await biGet('/analysis/api/bi/repeat/');
  const rows = data.rows || [];
  const foot = document.getElementById('repeatFoot');
  if (foot) {
    foot.textContent = '"New" means the first 30-day period in which a household appears inside '
      + 'the current selection, so filtering to a department reports households new to that '
      + 'department. ' + (data.note || '');
  }
  biDestroy(BI.repeat);
  BI.repeat = new Chart(canvas, {
    data: {
      labels: rows.map(r => 'P' + r.period),
      datasets: [
        {type: 'bar', label: 'Returning households', backgroundColor: T.indigo, borderRadius: 3,
         data: rows.map(r => Number(r.returning_revenue)), stack: 'rev', yAxisID: 'y'},
        {type: 'bar', label: 'New households', backgroundColor: T.amber, borderRadius: 3,
         data: rows.map(r => Number(r.new_revenue)), stack: 'rev', yAxisID: 'y'},
        {type: 'line', label: 'Returning share of revenue', borderColor: T.rose,
         backgroundColor: T.rose, data: rows.map(r => Number(r.returning_share)),
         yAxisID: 'y1', tension: .3, pointRadius: 2, borderWidth: 2},
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      onClick: (e, els) => {
        if (!els.length) return;
        biSetFilter('period', rows[els[0].index].period, {toggle: true});
      },
      scales: {
        y: {stacked: true, ticks: {callback: money}, title: {display: true, text: 'Revenue'}},
        y1: {position: 'right', min: 0, max: 100, grid: {drawOnChartArea: false},
             ticks: {callback: v => v + '%'}, title: {display: true, text: 'Returning share'}},
        x: {stacked: true},
      },
      plugins: {tooltip: {callbacks: {afterBody: items => {
        const row = rows[items[0].dataIndex];
        return row.is_first_period
          ? 'First period in the data: nothing precedes it, so every household counts as new.'
          : `${num(row.new_households)} new · ${num(row.returning_households)} returning`;
      }}}},
    },
  });
}

async function loadHouseholdValue() {
  const canvas = document.getElementById('householdChart');
  if (!canvas) return;
  const data = await biGet('/analysis/api/bi/household-value/');
  const rows = data.rows || [];
  const note = document.getElementById('hhValueNote');
  if (note && rows.length) {
    const topShare = Number(rows[0].revenue_share);
    const halfway = rows.find(r => Number(r.cumulative_share) >= 50);
    note.innerHTML = `Heaviest tenth = <strong>${topShare.toFixed(1)}%</strong> of revenue`
      + (halfway ? ` &middot; top ${halfway.decile * 10}% of households pass half of it` : '');
  }
  biDestroy(BI.household);
  BI.household = new Chart(canvas, {
    data: {
      labels: rows.map(r => 'Decile ' + r.decile),
      datasets: [
        {type: 'bar', label: 'Share of revenue', borderRadius: 4,
         backgroundColor: rows.map((_, i) => i === 0 ? T.amber : T.indigo),
         data: rows.map(r => Number(r.revenue_share)), yAxisID: 'y'},
        {type: 'line', label: 'Cumulative', borderColor: T.emerald, backgroundColor: T.emerald,
         data: rows.map(r => Number(r.cumulative_share)), yAxisID: 'y', tension: .3,
         pointRadius: 2, borderWidth: 2},
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {y: {ticks: {callback: v => v + '%'}, title: {display: true, text: 'Share of revenue'}}},
      plugins: {tooltip: {callbacks: {afterBody: items => {
        const row = rows[items[0].dataIndex];
        return [`${num(row.households)} households`,
                `Average spend ${money(row.avg_revenue)}`,
                `Average baskets ${Number(row.avg_baskets).toFixed(1)}`];
      }}}},
    },
  });
}

async function loadDiscountMix() {
  const canvas = document.getElementById('discountMixChart');
  if (!canvas) return;
  const data = await biGet('/analysis/api/bi/discount-mix/');
  const rows = data.rows || [];
  biDestroy(BI.discountMix);
  BI.discountMix = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: rows.map(r => r.band),
      datasets: [{
        label: 'Revenue', data: rows.map(r => Number(r.revenue)), borderRadius: 4,
        backgroundColor: [T.emerald, T.cyan, T.indigo, T.amber, T.rose],
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {y: {ticks: {callback: money}}},
      plugins: {
        legend: {display: false},
        tooltip: {callbacks: {
          label: c => money(c.parsed.y),
          afterBody: items => {
            const row = rows[items[0].dataIndex];
            return [`${Number(row.revenue_share).toFixed(1)}% of revenue`,
                    `${num(row.lines)} lines · ${num(row.coupon_lines)} used a coupon`];
          },
        }},
      },
    },
  });
}

async function loadBrandMix() {
  const canvas = document.getElementById('brandMixChart');
  if (!canvas) return;
  const data = await biGet('/analysis/api/bi/brand-mix/');
  const rows = (data.rows || []).slice(0, 14);
  biDestroy(BI.brandMix);
  BI.brandMix = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: rows.map(r => r.department),
      datasets: [
        {label: 'Private label', backgroundColor: T.violet, borderRadius: 3,
         data: rows.map(r => Number(r.private_share)), stack: 'mix'},
        {label: 'National brands', backgroundColor: T.cyan, borderRadius: 3,
         data: rows.map(r => Number(r.national_share)), stack: 'mix'},
      ],
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      onClick: (event, elements) => {
        if (!elements.length) return;
        biSetFilter('department', rows[elements[0].index].department, {toggle: true});
      },
      scales: {
        x: {stacked: true, max: 100, ticks: {callback: v => v + '%'}},
        y: {stacked: true},
      },
      plugins: {tooltip: {callbacks: {
        label: c => `${c.dataset.label}: ${Number(c.parsed.x).toFixed(1)}%`,
        afterBody: items => `Department revenue ${money(rows[items[0].dataIndex].revenue)}`,
      }}},
    },
  });
}

/* The heatmap is drawn as a grid rather than a chart: it needs one clickable
   cell per day-and-hour pair, and each cell sets both filters at once. */
async function loadHeatmap() {
  const grid = document.getElementById('heatmapGrid');
  if (!grid) return;
  const data = await biGet('/analysis/api/bi/heatmap/');
  const rows = data.rows || [];
  const peakBox = document.getElementById('heatmapPeak');
  if (peakBox) {
    peakBox.innerHTML = data.peak
      ? `Busiest: <strong>${escapeHtml(data.peak.day)} ${String(data.peak.hour).padStart(2, '0')}:00</strong>
         &middot; ${money(data.peak.revenue)}`
      : '';
  }
  if (!rows.length) { grid.innerHTML = '<div class="bi-loading">No trading hours in this selection.</div>'; return; }
  const days = [];
  rows.forEach(r => { if (!days.some(d => d.name === r.day_name)) days.push({name: r.day_name, sort: r.day_sort}); });
  days.sort((a, b) => a.day_sort - b.day_sort || a.sort - b.sort);
  const hours = [...new Set(rows.map(r => Number(r.hour)))].sort((a, b) => a - b);
  const byKey = new Map(rows.map(r => [`${r.day_name}|${Number(r.hour)}`, r]));
  const peak = Math.max(...rows.map(r => Number(r.revenue) || 0)) || 1;
  const cell = (value) => {
    const t = Math.pow((value || 0) / peak, 0.6);
    return `rgba(102,126,234,${(0.06 + t * 0.88).toFixed(3)})`;
  };
  grid.innerHTML = `<table><thead><tr><th></th>${
    hours.map(h => `<th>${String(h).padStart(2, '0')}</th>`).join('')
  }</tr></thead><tbody>${
    days.map(day => `<tr><th>${escapeHtml(day.name)}</th>${
      hours.map(hour => {
        const row = byKey.get(`${day.name}|${hour}`);
        const revenue = row ? Number(row.revenue) : 0;
        const selected = BI.filters.weekday === day.name && String(BI.filters.hour) === String(hour);
        return `<td class="${selected ? 'selected' : ''}" style="background:${cell(revenue)}"
                 title="${escapeHtml(day.name)} ${String(hour).padStart(2, '0')}:00 — ${money(revenue)}${row ? ' · ' + num(row.baskets) + ' baskets' : ''}"
                 onclick="biHeatmapPick('${escapeHtml(day.name)}', ${hour})"></td>`;
      }).join('')
    }</tr>`).join('')
  }</tbody></table>`;
}

/* One click sets both halves of the pair, so the cell and the filter chips
   always describe the same slice. */
function biHeatmapPick(day, hour) {
  biSetFilters([['weekday', day], ['hour', hour]], {toggle: true});
}

/* ---------- chart sizing -------------------------------------------------- */

const BI_LOADERS = [
  ['productChart',  () => BI.product,  loadProduct],
  ['timeChart',     () => BI.time,     loadTime],
  ['segmentChart',  () => BI.segment,  loadSegments],
  ['storeChart',    () => BI.store,    loadStores],
  ['basketChart',   () => BI.basket,   loadBaskets],
  ['hourChart',     () => BI.hour,     loadDaypart],
  ['weekdayChart',  () => BI.weekday,  loadDaypart],
  ['discountChart', () => BI.discount, loadDiscountTrend],
  ['brandChart',    () => BI.brand,    loadBrand],
  ['demoChart',     () => BI.demo,     loadDemographics],
  ['paretoChart',      () => BI.pareto,      loadPareto],
  ['growthChart',      () => BI.growth,      loadGrowth],
  ['repeatChart',      () => BI.repeat,      loadRepeat],
  ['householdChart',   () => BI.household,   loadHouseholdValue],
  ['discountMixChart', () => BI.discountMix, loadDiscountMix],
  ['brandMixChart',    () => BI.brandMix,    loadBrandMix],
];

async function biSettleCharts(attempt = 0) {
  const ratio = chartRenderPixelRatio();
  const rebuilds = new Set();
  BI_LOADERS.forEach(([id, get, load]) => {
    const chart = get();
    const canvas = document.getElementById(id);
    if (!chart || !canvas) return;
    const box = canvas.parentElement;
    if (!box || box.clientWidth < 1) return;   // page still hidden
    if (canvas.width < Math.floor(box.clientWidth * ratio) - 4) {
      /* The bitmap is smaller than its box. resize() cannot fix this: Chart.js
         compares against its own record, which already matches the box, so the
         call is a no-op while the canvas stays small and renders stretched.
         Rebuilding measures the box afresh. */
      rebuilds.add(load);
    }
    if (box.dataset.biObserved) return;
    box.dataset.biObserved = '1';
    new ResizeObserver(() => get()?.resize()).observe(box);
  });
  if (rebuilds.size) {
    await Promise.all([...rebuilds].map(load => load()));
    biRefreshDataViews();
  }
  if (attempt < 4) setTimeout(() => biSettleCharts(attempt + 1), 300);
}

/* ---------- per-visual controls ------------------------------------------
   The three things a reader reaches for on any tile: make it bigger, see the
   numbers behind it, take it away. Attached to every card from the chart
   registry, so a new panel gets them by being registered rather than by
   repeating markup. */

const BI_CHART_BY_ID = () => Object.fromEntries(BI_LOADERS.map(([id, get]) => [id, get]));

function biCardTitle(card) {
  return (card.querySelector('.bi-head h5')?.textContent || 'panel').trim();
}

function biDecorateCards() {
  document.querySelectorAll('.bi-card').forEach(card => {
    if (card.dataset.biTools) return;
    const canvas = card.querySelector('canvas');
    const grid = card.querySelector('.bi-heatmap');
    if (!canvas && !grid) return;             // tables already read as data
    card.dataset.biTools = '1';
    const tools = document.createElement('div');
    tools.className = 'bi-vis-tools';
    tools.innerHTML = `
      <button type="button" title="Show the numbers behind this" data-act="table"><i class="fas fa-table-list"></i></button>
      <button type="button" title="Download as CSV" data-act="csv"><i class="fas fa-download"></i></button>
      <button type="button" title="Focus this panel" data-act="focus"><i class="fas fa-expand"></i></button>`;
    tools.addEventListener('click', event => {
      const button = event.target.closest('button');
      if (!button) return;
      const act = button.dataset.act;
      if (act === 'table') biToggleDataView(card, button);
      if (act === 'csv') biExportCard(card);
      if (act === 'focus') biToggleFocus(card, button);
    });
    card.appendChild(tools);
  });
}

/* Rows behind a card, taken from the chart that is already on screen so the
   table can never disagree with the picture. */
function biCardRows(card) {
  const canvas = card.querySelector('canvas');
  if (canvas) {
    const chart = (BI_CHART_BY_ID()[canvas.id] || (() => null))();
    if (!chart) return null;
    const sets = chart.data.datasets || [];
    // Chart.js stores an empty array when a chart is plotted from {x, y} points
    // rather than labels, so an empty one has to count as "no labels" or the
    // table comes out with headings and nothing under them.
    const labels = (chart.data.labels && chart.data.labels.length) ? chart.data.labels : null;
    const length = labels ? labels.length
      : Math.max(0, ...sets.map(d => (d.data || []).length));
    const firstHeading = labels
      ? 'Category'
      : (chart.options?.scales?.x?.title?.text || 'Point');
    const header = [firstHeading, ...sets.map(d => d.label || 'Value')];
    const body = [];
    for (let i = 0; i < length; i++) {
      const lead = sets[0]?.data?.[i];
      const first = labels ? labels[i]
        : (lead && typeof lead === 'object' ? (lead.x ?? i + 1) : i + 1);
      body.push([first, ...sets.map(d => {
        const point = (d.data || [])[i];
        if (point && typeof point === 'object') return point.y ?? point.r ?? '';
        return point ?? '';
      })]);
    }
    return {header, body};
  }
  const table = card.querySelector('.bi-heatmap table');
  if (!table) return null;
  const hours = [...table.querySelectorAll('thead th')].slice(1).map(th => th.textContent.trim());
  const body = [...table.querySelectorAll('tbody tr')].map(tr => [
    tr.querySelector('th').textContent.trim(),
    ...[...tr.querySelectorAll('td')].map(td => {
      const match = (td.getAttribute('title') || '').match(/\u2014\s*\$([\d,]+)/);
      return match ? match[1].replace(/,/g, '') : '';
    }),
  ]);
  return {header: ['Day', ...hours], body};
}

function biToggleDataView(card, button) {
  const open = card.querySelector('.bi-datatable');
  if (open) {
    open.remove();
    delete card.dataset.biDataview;
    button.classList.remove('on');
    return;
  }
  card.dataset.biDataview = '1';
  biRenderDataView(card);
  button.classList.add('on');
}

/* An open table has to follow its chart. Drilling or cross-filtering rebuilds
   the chart, and a table left as first drawn would keep quoting the level the
   reader has already moved off. */
function biRefreshDataViews() {
  document.querySelectorAll('.bi-card[data-bi-dataview]').forEach(biRenderDataView);
}

function biRenderDataView(card) {
  card.querySelector('.bi-datatable')?.remove();
  const rows = biCardRows(card);
  if (!rows) return;
  /* Sorting is applied to the table rather than to the chart axis: several
     charts resolve a click to a filter by the mark's position, so reordering
     the plotted data would send clicks to the wrong category. */
  const sortColumn = card.dataset.biSortCol === undefined ? null : Number(card.dataset.biSortCol);
  const descending = card.dataset.biSortDir !== 'asc';
  if (sortColumn !== null) {
    const numeric = v => (v !== '' && v !== null && !isNaN(Number(v)));
    rows.body = rows.body.slice().sort((a, b) => {
      const x = a[sortColumn], y = b[sortColumn];
      const both = numeric(x) && numeric(y);
      const order = both ? Number(x) - Number(y) : String(x).localeCompare(String(y));
      return descending ? -order : order;
    });
  }
  const isNumber = v => v !== '' && v !== null && !isNaN(Number(v));
  const box = document.createElement('div');
  box.className = 'bi-datatable';
  box.innerHTML = `<table><thead><tr>${
    rows.header.map((h, i) => {
      const active = sortColumn === i;
      const arrow = active ? (descending ? ' ▾' : ' ▴') : '';
      return `<th class="${i ? 'n' : ''} bi-sortable" data-col="${i}"
                title="Sort by this column">${escapeHtml(h)}${arrow}</th>`;
    }).join('')
  }</tr></thead><tbody>${
    rows.body.map(row => `<tr>${row.map((cell, i) => {
      /* The first column is a category. Whole numbers there are identities --
         a year or a store id -- so they keep their digits; a fraction there is
         a plotted position and only needs a few figures. */
      let value = cell;
      if (isNumber(cell)) {
        const n = Number(cell);
        if (!i) value = Number.isInteger(n) ? String(n)
          : n.toLocaleString(undefined, {maximumSignificantDigits: 4});
        else value = n.toLocaleString(undefined, {maximumFractionDigits: 2});
      }
      return `<td class="${i ? 'n' : ''}">${escapeHtml(value)}</td>`;
    }).join('')}</tr>`).join('')
  }</tbody></table>`;
  box.querySelectorAll('th.bi-sortable').forEach(th => {
    th.addEventListener('click', () => {
      const column = Number(th.dataset.col);
      if (Number(card.dataset.biSortCol) === column) {
        card.dataset.biSortDir = card.dataset.biSortDir === 'asc' ? 'desc' : 'asc';
      } else {
        card.dataset.biSortCol = String(column);
        card.dataset.biSortDir = column ? 'desc' : 'asc';   // numbers high first, names A-Z
      }
      biRenderDataView(card);
    });
  });
  card.appendChild(box);
}

function biExportCard(card) {
  const rows = biCardRows(card);
  if (!rows) return;
  const quote = v => {
    const text = String(v ?? '');
    return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  };
  const csv = [rows.header, ...rows.body].map(row => row.map(quote).join(',')).join('\n');
  const name = biCardTitle(card).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
  const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${name || 'panel'}.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

/* Focus lifts one card over a backdrop. The chart is rebuilt afterwards rather
   than resized, for the same reason the tab switch rebuilds: a canvas measured
   at one size will not stretch cleanly to another. */
function biToggleFocus(card, button) {
  const layer = document.querySelector('.bi-focus-layer');
  if (layer && layer.contains(card)) { biCloseFocus(); return; }
  if (layer) biCloseFocus();
  // Leave a marker so the card returns to exactly where it came from.
  const placeholder = document.createElement('div');
  placeholder.dataset.biPlaceholder = '1';
  card.parentNode.insertBefore(placeholder, card);
  const host = document.createElement('div');
  host.className = 'bi-focus-layer';
  host.addEventListener('click', event => { if (event.target === host) biCloseFocus(); });
  host.appendChild(card);
  document.body.appendChild(host);
  document.body.style.overflow = 'hidden';
  button?.classList.add('on');
  setTimeout(() => biSettleCharts(), 80);
}

function biCloseFocus() {
  const host = document.querySelector('.bi-focus-layer');
  if (!host) return;
  const card = host.querySelector('.bi-card');
  const placeholder = document.querySelector('[data-bi-placeholder]');
  if (card && placeholder) {
    placeholder.parentNode.insertBefore(card, placeholder);
    placeholder.remove();
    card.querySelector('.bi-vis-tools button[data-act="focus"]')?.classList.remove('on');
  }
  host.remove();
  document.body.style.overflow = '';
  setTimeout(() => biSettleCharts(), 80);
}

document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  biCloseFocus();
});

/* ---------- saved views --------------------------------------------------- */

const BI_BOOKMARK_KEY = 'bi.bookmarks.v1';

function biBookmarks() {
  try { return JSON.parse(localStorage.getItem(BI_BOOKMARK_KEY) || '[]'); }
  catch (error) { return []; }
}

function biSaveBookmark() {
  const keys = Object.keys(BI.filters);
  if (!keys.length) { showNotification('Set some filters first, then save the view.', 'info'); return; }
  const suggested = keys.map(k => String(BI.filters[k]).split('|').join(', ')).join(' · ');
  const name = window.prompt('Name this view', suggested.slice(0, 60));
  if (!name) return;
  const all = biBookmarks().filter(b => b.name !== name);
  all.push({name, filters: {...BI.filters}, page: BI.page});
  localStorage.setItem(BI_BOOKMARK_KEY, JSON.stringify(all.slice(-12)));
  renderBookmarks();
}

function biApplyBookmark(name) {
  const found = biBookmarks().find(b => b.name === name);
  if (!found) return;
  BI.filters = {...found.filters};
  biSyncSlicers();
  biRefresh();
}

function biDeleteBookmark(name, event) {
  event.stopPropagation();
  localStorage.setItem(BI_BOOKMARK_KEY, JSON.stringify(biBookmarks().filter(b => b.name !== name)));
  renderBookmarks();
}

function renderBookmarks() {
  const box = document.getElementById('biBookmarks');
  if (!box) return;
  const all = biBookmarks();
  box.innerHTML = all.length
    ? '<span class="bi-stamp me-1">Saved views</span>' + all.map(b => `
        <button class="bi-bookmark" type="button" onclick="biApplyBookmark('${escapeHtml(b.name).replace(/'/g, "\\'")}')">
          <i class="fas fa-bookmark"></i><b>${escapeHtml(b.name)}</b>
          <span class="x" title="Delete"
                onclick="biDeleteBookmark('${escapeHtml(b.name).replace(/'/g, "\\'")}', event)">&times;</span>
        </button>`).join('')
    : '';
}

/* ---------- orchestration ------------------------------------------------- */

async function biRefresh() {
  const run = ++biRun;
  renderChips();
  try {
    const kpis = await biGet('/analysis/api/bi/kpis/');
    renderKpis(kpis.kpis);
    await Promise.all([
      loadProduct(), loadTime(), loadSegments(), loadStores(), loadBaskets(),
      loadDaypart(), loadDiscountTrend(), loadBrand(), loadDemographics(),
      loadTopProducts(), loadInsights(), loadSignificance(),
      loadPareto(), loadGrowth(), loadRepeat(), loadHouseholdValue(),
      loadDiscountMix(), loadBrandMix(), loadHeatmap(),
      biScanRequested ? loadSignificanceScan() : null,
    ]);
    setTimeout(() => { biSettleCharts(); biDecorateCards(); biRefreshDataViews(); }, 250);
    const stamp = document.getElementById('biStamp');
    if (stamp) {
      stamp.textContent = 'Read from the database at ' + new Date().toLocaleTimeString()
        + '. Ctrl-click a mark to add it to the selection.';
    }
    renderBookmarks();
  } catch (error) {
    // A refresh the user has already replaced is not a failure worth reporting.
    if (error === BI_STALE || run !== biRun) return;
    console.error(error);
    showNotification('Dashboard could not load: ' + error.message, 'danger');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const tab = document.getElementById('kpiStrip')?.closest('.tab-content');
  // Charts built inside a hidden tab measure a zero-width canvas, so hold the
  // first load until the tab is shown. The standalone page has no tab.
  if (!tab || tab.classList.contains('active')) { biRefresh(); return; }
  const observer = new MutationObserver(() => {
    if (!tab.classList.contains('active')) return;
    observer.disconnect();
    biRefresh();
  });
  observer.observe(tab, {attributes: true, attributeFilter: ['class']});
}, {once: true});
