
document.addEventListener('DOMContentLoaded', () => {
  const ruleCards = document.querySelectorAll('.rule-card');

  ruleCards.forEach((card) => {
    const insertButton = card.querySelector('.insert-rule-btn');
    if (insertButton) {
      insertButton.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        const ruleData = buildRuleData(card);
        openInsertConfirm(ruleData, card);
      });
    }

    card.addEventListener('click', () => {
      const ruleData = buildRuleData(card);
      const title = `🔗 Association Rule: ${card.dataset.ant} → ${card.dataset.con}`;
      const body = `
        <div class="row g-3">
          <div class="col-12">
            <div class="alert alert-info">
              <h6><i class="fas fa-info-circle"></i> Rule Interpretation</h6>
              <p class="mb-0">When customers buy product <strong>${card.dataset.ant}</strong>, they also tend to buy product <strong>${card.dataset.con}</strong></p>
            </div>
          </div>
          <div class="col-md-4">
            <div class="card border-primary">
              <div class="card-body text-center">
                <div class="text-primary h4">${card.dataset.sup}</div>
                <div class="text-muted small">Support</div>
                <small class="text-muted d-block">Frequency of both items together</small>
              </div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="card border-warning">
              <div class="card-body text-center">
                <div class="text-warning h4">${card.dataset.conf}</div>
                <div class="text-muted small">Confidence</div>
                <small class="text-muted d-block">Reliability of the rule</small>
              </div>
            </div>
          </div>
          <div class="col-md-4">
            <div class="card border-success">
              <div class="card-body text-center">
                <div class="text-success h4">${card.dataset.lift}</div>
                <div class="text-muted small">Lift</div>
                <small class="text-muted d-block">Strength of association</small>
              </div>
            </div>
          </div>
          <div class="col-12">
            <div class="bg-light p-3 rounded">
              <h6><i class="fas fa-lightbulb text-warning"></i> Business Insights</h6>
              <ul class="mb-0">
                <li><strong>Cross-selling opportunity:</strong> Promote product ${card.dataset.con} to customers buying ${card.dataset.ant}</li>
                <li><strong>Store layout:</strong> Consider placing these products near each other</li>
                <li><strong>Bundle pricing:</strong> Create attractive package deals for these items</li>
              </ul>
            </div>
          </div>
          <div class="col-12 mt-3">
            <div class="text-end">
              <button type="button" class="btn btn-success" id="modalInsertRuleBtn">
                <i class="fas fa-database"></i> Insert This Rule
              </button>
            </div>
          </div>
        </div>
      `;
      showDetailModal(title, body);
      setTimeout(() => {
        const modalInsertBtn = document.getElementById('modalInsertRuleBtn');
        if (modalInsertBtn) {
          modalInsertBtn.addEventListener('click', (event) => {
            event.preventDefault();
            // The same card the modal was opened from, so inserting here marks
            // it saved too rather than only the button on the card itself.
            openInsertConfirm(ruleData, card);
          }, { once: true });
        }
      }, 200);
    });
  });
});

function buildRuleData(card) {
  const splitList = (value) => {
    if (!value) {
      return [];
    }
    const delimiter = value.indexOf('|') !== -1 ? '|' : (value.indexOf(',') !== -1 ? ',' : null);
    if (!delimiter) {
      return [value.trim()];
    }
    return value.split(delimiter).map((item) => item.trim()).filter(Boolean);
  };

  const parseNumber = (value) => {
    const num = parseFloat(value);
    return Number.isFinite(num) ? num : null;
  };

  const antecedent = splitList(card.dataset.antList || card.dataset.ant);
  const consequent = splitList(card.dataset.conList || card.dataset.con);
  const support = parseNumber(card.dataset.sup);
  const confidence = parseNumber(card.dataset.conf);
  const lift = parseNumber(card.dataset.lift);

  let ruleType = card.dataset.type || 'product';
  if (['product', 'category', 'commodity', 'department'].indexOf(ruleType) === -1) {
    ruleType = 'product';
  }

  const minSupportValue = parseNumber(card.dataset.minSupport);
  const minConfidenceValue = parseNumber(card.dataset.minConfidence);
  const minLiftValue = parseNumber(card.dataset.minLift);

  return {
    antecedent: antecedent,
    consequent: consequent,
    support: support,
    confidence: confidence,
    lift: lift,
    rule_type: ruleType,
    min_support_threshold: minSupportValue !== null ? minSupportValue : support,
    min_confidence_threshold: minConfidenceValue,
    min_lift_threshold: minLiftValue,
    source_view: card.dataset.source || 'analysis.association_rules',
    metadata: {
      antecedent_label: card.dataset.ant,
      consequent_label: card.dataset.con,
      source_path: window.location.pathname
    }
  };
}

/* A card went on offering "Insert Rule" after its rule had been stored, and
   only a page reload would say otherwise. That made the same rule easy to
   insert twice and gave no sign the first attempt had worked. The card is
   brought up to date in place instead, carrying the same badge the server
   renders for an already-saved rule on the next load. */
function markRuleCardSaved(card) {
  if (!card || card.dataset.savedState === 'saved') return;
  card.dataset.savedState = 'saved';

  const stats = card.querySelector('.border-top');
  if (stats) {
    // "Update available" no longer applies once the saved copy matches this one.
    stats.querySelectorAll('.ar-saved, .ar-changed').forEach(chip => chip.remove());
    const badge = document.createElement('div');
    badge.className = 'ar-saved';
    badge.title = 'Stored from this page a moment ago';
    /* Written out rather than left to toLocaleDateString, so it matches what
       Django's "j M Y" renders for this same badge on the next load. The
       locale versions disagree with it and with each other: the default gives
       "Sep 6, 2026" and en-GB gives "6 Sept 2026", so the one card would be
       dated differently before and after a refresh. */
    const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const now = new Date();
    const when = `${now.getDate()} ${MONTHS[now.getMonth()]} ${now.getFullYear()}`;
    badge.innerHTML = `<i class="fas fa-database"></i> Saved `
      + `<span class="ar-saved-date">${when}</span>`;
    stats.insertBefore(badge, stats.firstChild);
  }

  const button = card.querySelector('.insert-rule-btn');
  if (button) {
    button.classList.remove('btn-outline-success');
    button.classList.add('btn-outline-secondary');
    button.innerHTML = '<i class="fas fa-check"></i> Saved';
    button.disabled = true;
  }

  /* Leaving it ticked would send it again with the next bulk insert, for no
     effect but a wasted request. */
  const checkbox = card.querySelector('.rule-checkbox');
  if (checkbox && checkbox.checked) {
    checkbox.checked = false;
    if (typeof updateSelectedRulesCount === 'function') updateSelectedRulesCount();
  }
}

function openInsertConfirm(ruleData, card) {
  /* Four decimal places suit confidence and lift, but not support, which on
     this data runs into the ten-thousandths: 0.000130 was shown as 0.0001, so
     the figure the reader confirmed was not the figure being stored. Values
     below 0.001 are given three significant figures instead, in plain notation
     rather than an exponent. */
  const formatMetric = (value) => {
    if (!Number.isFinite(value)) return 'N/A';
    if (value === 0) return '0';
    if (Math.abs(value) >= 0.001) return value.toFixed(4);
    const places = 2 - Math.floor(Math.log10(Math.abs(value)));
    return value.toFixed(Math.min(places, 100));
  };

  const body = `
    <div class="alert alert-warning">
      <i class="fas fa-database me-2"></i>
      Store the rule <strong>${ruleData.antecedent.join(', ')}</strong> → <strong>${ruleData.consequent.join(', ')}</strong> for future recommendations?
    </div>
    <ul class="list-unstyled small mb-3">
      <li><strong>Support:</strong> ${formatMetric(ruleData.support)}</li>
      <li><strong>Confidence:</strong> ${formatMetric(ruleData.confidence)}</li>
      <li><strong>Lift:</strong> ${formatMetric(ruleData.lift)}</li>
    </ul>
    <div class="text-end">
      <button type="button" class="btn btn-secondary me-2" data-bs-dismiss="modal">Cancel</button>
      <button type="button" class="btn btn-success" id="confirmInsertRuleBtn">
        <i class="fas fa-database"></i> Insert Rule
      </button>
    </div>
  `;
  showDetailModal('Confirm Association Rule Insert', body);
  setTimeout(() => {
    const confirmBtn = document.getElementById('confirmInsertRuleBtn');
    if (confirmBtn) {
      confirmBtn.addEventListener('click', () => performRuleInsert(ruleData, card), { once: true });
    }
  }, 200);
}

function performRuleInsert(ruleData, card) {
  const modalEl = document.getElementById('detailModal');
  const modalInstance = modalEl ? bootstrap.Modal.getInstance(modalEl) : null;
  const confirmBtn = document.getElementById('confirmInsertRuleBtn');
  if (confirmBtn) {
    confirmBtn.disabled = true;
    confirmBtn.classList.add('disabled');
  }

  fetch('/analysis/api/association-rules/insert/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCsrfToken()
    },
    body: JSON.stringify(ruleData)
  })
    .then(async (response) => {
      const payload = await response.json().catch(() => ({ success: false, error: 'Unexpected response from server.' }));
      if (response.ok && payload.success) {
        if (modalInstance) {
          modalInstance.hide();
        }
        markRuleCardSaved(card);
        showRuleInsertFeedback(payload.message || 'Rule stored successfully.', 'success');
      } else {
        const errorMessage = payload.error || 'Unable to store the rule.';
        showRuleInsertFeedback(errorMessage, 'danger');
        if (modalInstance) {
          modalInstance.hide();
        }
      }
    })
    .catch((error) => {
      console.error('Failed to insert association rule', error);
      showRuleInsertFeedback('Unexpected error while saving the rule.', 'danger');
      if (modalInstance) {
        modalInstance.hide();
      }
    })
    .finally(() => {
      if (confirmBtn) {
        confirmBtn.disabled = false;
        confirmBtn.classList.remove('disabled');
      }
    });
}

function showRuleInsertFeedback(message, variant) {
  const container = document.getElementById('ruleInsertAlert');
  if (!container) {
    alert(message);
    return;
  }
  container.textContent = message;
  container.className = `alert alert-${variant}`;
  container.classList.remove('d-none');
  setTimeout(() => {
    container.classList.add('d-none');
  }, 5000);
}

function getCsrfToken() {
  const match = document.cookie.match(/csrftoken=([^;]+)/);
  return match ? match[1] : '';
}

// Loading functionality for association rules generation
document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('associationRulesForm');
  const generateBtn = document.getElementById('generateBtn');
  const loadingState = document.getElementById('loadingState');
  const resultsContainer = document.getElementById('resultsContainer');
  const periodSelect = document.getElementById('transaction_period');
  const periodText = document.getElementById('periodText');

  // Update period text when selection changes
  if (periodSelect && periodText) {
    periodSelect.addEventListener('change', function() {
      updatePeriodText();
    });
    updatePeriodText(); // Set initial text
  }

  function updatePeriodText() {
    const selectedOption = periodSelect.options[periodSelect.selectedIndex];
    periodText.textContent = selectedOption.text.toLowerCase();
  }

  // Handle form submission with loading state
  if (form && generateBtn && loadingState && resultsContainer) {
    form.addEventListener('submit', function(e) {
      // Show loading state
      generateBtn.disabled = true;
      generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

      loadingState.classList.remove('d-none');
      resultsContainer.style.display = 'none';

      // Allow form to submit normally
      // The page will reload with results or error message
    });
  }
});

// Real-time support value calculator
/* Five thousand rules is a legitimate request, but laying them all out at once
   made a three-million-pixel page: one scroll to the bottom cost 1.6 seconds.
   The cards are all here in the document; only a page of them is displayed, so
   the browser lays out a hundred instead of five thousand. Selection and the
   insert count still work across the whole set. */
const RULE_PAGE_SIZE = 100;
let rulePage = 1;

function ruleCells() {
  const grid = document.getElementById('ruleGrid');
  return grid ? Array.from(grid.children) : [];
}

/* The pager walks the cards that match the current search and filters, not
   every card in the grid, so "Showing 1-100 of 2,000" keeps telling the truth
   once a filter is on. A cell that does not match is hidden outright and never
   enters the page arithmetic. */
function ruleFilterState() {
  const term = (document.getElementById('ruleSearch')?.value || '').trim().toLowerCase();
  return {
    term,
    department: (document.getElementById('ruleDepartmentFilter')?.value || '').toLowerCase(),
    lift: document.getElementById('ruleLiftFilter')?.value || '',
    active: Boolean(term) || Boolean(document.getElementById('ruleDepartmentFilter')?.value)
      || Boolean(document.getElementById('ruleLiftFilter')?.value),
  };
}

function ruleCellMatches(cell, state) {
  const card = cell.querySelector('.rule-card');
  if (!card) return true;
  if (state.term && !(card.dataset.search || '').includes(state.term)) return false;
  if (state.department && !(card.dataset.departments || '').includes(state.department + '|')) return false;
  if (state.lift) {
    const lift = Number(card.dataset.lift);
    if (!Number.isFinite(lift)) return false;
    if (state.lift === 'above' && !(lift >= 1)) return false;
    if (state.lift === 'below' && !(lift < 1)) return false;
  }
  return true;
}

function matchingRuleCells() {
  const state = ruleFilterState();
  if (!state.active) return ruleCells();
  return ruleCells().filter(cell => ruleCellMatches(cell, state));
}

function applyRuleFilter() {
  rulePage = 1;
  const state = ruleFilterState();
  const clear = document.getElementById('ruleFilterClear');
  if (clear) clear.hidden = !state.active;
  renderRulePage();
}

function clearRuleFilter() {
  const search = document.getElementById('ruleSearch');
  const department = document.getElementById('ruleDepartmentFilter');
  const lift = document.getElementById('ruleLiftFilter');
  if (search) search.value = '';
  if (department) department.value = '';
  if (lift) lift.value = '';
  applyRuleFilter();
}

/* The list of departments comes from the rules on the page rather than the 43
   in the product table, so it never offers a department that would return
   nothing. The count beside each name says so plainly: without it the list
   looks like it ought to be the whole catalogue and its shortness reads as a
   bug rather than as the point. */
function buildDepartmentFilter() {
  const select = document.getElementById('ruleDepartmentFilter');
  if (!select) return;
  const seen = new Map();
  ruleCells().forEach(cell => {
    const card = cell.querySelector('.rule-card');
    if (!card) return;
    (card.dataset.departments || '').split('|').forEach(name => {
      const trimmed = name.trim();
      if (trimmed) seen.set(trimmed, (seen.get(trimmed) || 0) + 1);
    });
  });
  [...seen.entries()].sort((first, second) => first[0].localeCompare(second[0]))
    .forEach(([name, count]) => {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = `${name.toUpperCase()} (${count})`;
      select.appendChild(option);
    });
  const label = document.getElementById('ruleDepartmentHint');
  if (label) {
    label.textContent = `${seen.size} of 43 departments appear in these rules`;
  }
}

function renderRulePage() {
  const all = ruleCells();
  const cells = matchingRuleCells();
  const empty = document.getElementById('ruleNoMatches');
  if (empty) empty.hidden = cells.length > 0;
  if (!cells.length) {
    all.forEach(cell => { cell.style.display = 'none'; });
    ['rulePagerTop', 'rulePagerBottom'].forEach(id => {
      const box = document.getElementById(id);
      if (box) box.innerHTML = '';
    });
    return;
  }
  // Anything filtered out is hidden before the page window is applied.
  const matching = new Set(cells);
  all.forEach(cell => {
    if (!matching.has(cell)) cell.style.display = 'none';
  });
  const pages = Math.max(1, Math.ceil(cells.length / RULE_PAGE_SIZE));
  rulePage = Math.min(Math.max(rulePage, 1), pages);
  const first = (rulePage - 1) * RULE_PAGE_SIZE;
  cells.forEach((cell, index) => {
    cell.style.display = (index >= first && index < first + RULE_PAGE_SIZE) ? '' : 'none';
  });
  const shown = `${first + 1}-${Math.min(first + RULE_PAGE_SIZE, cells.length)}`;
  const controls = `
    <button type="button" class="pager-btn"
            onclick="goRulePage(1)" ${rulePage <= 1 ? 'disabled' : ''}>&laquo;&laquo; First</button>
    <button type="button" class="pager-btn"
            onclick="changeRulePage(-1)" ${rulePage <= 1 ? 'disabled' : ''}>Previous</button>
    <span class="pager-status">Showing <b>${shown}</b> of
      <b>${cells.length.toLocaleString()}</b> rules &middot; page ${rulePage} of ${pages}</span>
    <button type="button" class="pager-btn"
            onclick="changeRulePage(1)" ${rulePage >= pages ? 'disabled' : ''}>Next</button>
    <button type="button" class="pager-btn"
            onclick="goRulePage(0)" ${rulePage >= pages ? 'disabled' : ''}>Last &raquo;&raquo;</button>`;
  ['rulePagerTop', 'rulePagerBottom'].forEach(id => {
    const box = document.getElementById(id);
    if (box) box.innerHTML = cells.length > RULE_PAGE_SIZE ? controls : '';
  });
}

/* 0 means the last page, whatever it turns out to be. */
function goRulePage(page) {
  const pages = Math.max(1, Math.ceil(matchingRuleCells().length / RULE_PAGE_SIZE));
  rulePage = page === 0 ? pages : page;
  renderRulePage();
  document.getElementById('rulePagerTop')?.scrollIntoView({behavior: 'smooth', block: 'start'});
}

function changeRulePage(step) {
  rulePage += step;
  renderRulePage();
  document.getElementById('rulePagerTop')?.scrollIntoView({behavior: 'smooth', block: 'start'});
}

document.addEventListener('DOMContentLoaded', () => {
  buildDepartmentFilter();
  renderRulePage();
});

/* Basket counts per period, read from the database rather than written in.
   null means "not fetched yet"; the request is fired once and the answer kept. */
const basketCountCache = {};
const basketCountPending = {};

function seedBasketCountCache() {
  const badge = document.getElementById('datasetScaleBadge');
  const allTime = badge ? Number(badge.dataset.baskets) : 0;
  if (Number.isFinite(allTime) && allTime > 0) basketCountCache['all'] = allTime;
}

function basketsForPeriod(period) {
  if (basketCountCache[period] != null) return basketCountCache[period];
  if (!basketCountPending[period]) {
    basketCountPending[period] = true;
    const body = new URLSearchParams({transaction_period: period});
    fetch('/analysis/api/period-metrics/', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded', 'X-CSRFToken': getCsrfToken()},
      body: body.toString(),
    })
      .then(response => response.json())
      .then(payload => {
        const baskets = payload && payload.metrics && payload.metrics.total_baskets;
        if (Number.isFinite(Number(baskets))) basketCountCache[period] = Number(baskets);
        updateSupportInfo();
      })
      .catch(error => {
        console.error('Could not read basket counts for', period, error);
        // Falling back to the all-time figure is better than leaving the hint
        // blank; it is the same order of magnitude and the label says so.
        if (basketCountCache['all'] != null) basketCountCache[period] = basketCountCache['all'];
        updateSupportInfo();
      });
  }
  return null;
}

function updateSupportInfo() {
  const supportInput = document.getElementById('min_support');
  const countSpan = document.getElementById('currentSupportCount');
  const totalSpan = document.getElementById('supportBasketTotal');
  const periodSelect = document.getElementById('transaction_period');
  if (!supportInput || !countSpan) return;

  /* Support is measured against baskets, not transaction lines. This read
     transaction counts and so overstated every threshold about ninefold: 0.00001
     was labelled "26+ transactions" when the algorithm was in fact accepting
     pairs seen in 3 baskets.

     The five basket counts used to be written in here, correct only while the
     table did not change. They are fetched from the period-metrics endpoint
     instead and remembered per period, so typing in the support box does not
     put a request on the wire for every keystroke. The all-time count rendered
     into the banner seeds the cache, so the first paint needs no request. */
  const period = periodSelect ? periodSelect.value : 'all';
  const total = basketsForPeriod(period);
  if (total === null) {
    if (totalSpan) totalSpan.textContent = '…';
    countSpan.textContent = '…';
    return;
  }
  const support = parseFloat(supportInput.value);
  if (totalSpan) totalSpan.textContent = total.toLocaleString();
  if (isNaN(support) || support <= 0) { countSpan.textContent = '--'; return; }

  // Rounded up, matching the threshold the query applies.
  const needed = Math.max(1, Math.ceil(total * support));
  countSpan.textContent = needed.toLocaleString();
  const box = document.getElementById('supportInfo');
  if (box) {
    box.classList.toggle('is-thin', needed < 20);
    box.title = needed < 20
      ? 'Fewer than 20 baskets is thin evidence: a high lift on a handful of baskets is usually noise.'
      : '';
  }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
  seedBasketCountCache();
  updateSupportInfo();

  // Also update when period changes
  const periodSelect = document.getElementById('transaction_period');
  if (periodSelect) {
    periodSelect.addEventListener('change', updateSupportInfo);
  }

  // Initialize multi-select functionality
  initializeMultiSelect();
});

// Multi-select functionality for association rules
function selectAllRules(selectAll) {
  const checkboxes = document.querySelectorAll('.rule-checkbox');
  checkboxes.forEach(checkbox => {
    checkbox.checked = selectAll;
  });
  updateSelectedRulesCount();
}

function updateSelectedRulesCount() {
  const selectedCount = document.querySelectorAll('.rule-checkbox:checked').length;
  const totalCount = document.querySelectorAll('.rule-checkbox').length;

  const countElement = document.getElementById('selectedRulesCount');
  const actionsElement = document.getElementById('rulesMultiSelectActions');

  if (countElement) {
    countElement.textContent = selectedCount;
  }

  // Show/hide actions based on whether there are rules
  if (actionsElement) {
    actionsElement.style.display = totalCount > 0 ? 'flex' : 'none';
  }

  // Update insert button state
  const insertBtn = document.querySelector('#rulesMultiSelectActions .btn-success');
  if (insertBtn) {
    insertBtn.disabled = selectedCount === 0;
    if (selectedCount === 0) {
      insertBtn.classList.add('disabled');
    } else {
      insertBtn.classList.remove('disabled');
    }
  }

  // Update visual feedback for selected cards
  document.querySelectorAll('.selectable-rule').forEach(card => {
    const checkbox = card.querySelector('.rule-checkbox');
    if (checkbox && checkbox.checked) {
      card.style.background = 'linear-gradient(135deg, #e8f5e8, #f0f8f0)';
      card.style.transform = 'translateX(2px)';
    } else {
      card.style.background = '';
      card.style.transform = '';
    }
  });
}

function insertSelectedRules() {
  const selectedCards = document.querySelectorAll('.selectable-rule:has(.rule-checkbox:checked)');

  if (selectedCards.length === 0) {
    alert('Please select at least one rule to insert.');
    return;
  }

  const selectedRules = [];
  const duplicateRules = new Set();

  selectedCards.forEach(card => {
    const ruleData = {
      antecedent: card.dataset.ant,
      consequent: card.dataset.con,
      support: parseFloat(card.dataset.sup),
      confidence: parseFloat(card.dataset.conf),
      lift: parseFloat(card.dataset.lift),
      rule_type: card.dataset.type,
      /* Carried so each card can be marked saved as its own insert lands.
         insertSingleRule rebuilds its payload from named fields, so this
         element never reaches JSON.stringify. */
      card: card,
    };

    // Create unique identifier for duplicate detection
    const ruleId = `${ruleData.antecedent}→${ruleData.consequent}`;

    if (!duplicateRules.has(ruleId)) {
      duplicateRules.add(ruleId);
      selectedRules.push(ruleData);
    }
  });

  // Show confirmation dialog with filtered count
  const confirmMsg = `Insert ${selectedRules.length} selected association rules?${selectedRules.length < selectedCards.length ? ` (${selectedCards.length - selectedRules.length} duplicates removed)` : ''}`;
  if (!confirm(confirmMsg)) return;

  // Process bulk insertion with pre-validation
  processBulkRuleInsertion(selectedRules);
}

function processBulkRuleInsertion(rules) {
  let successCount = 0;
  let errorCount = 0;
  const errors = [];

  // Pre-validate rules
  const validRules = rules.filter(rule => validateRule(rule));
  const invalidCount = rules.length - validRules.length;

  if (invalidCount > 0) {
    console.log(`Filtered out ${invalidCount} invalid rules before insertion`);
  }

  // Show progress
  const alertElement = document.getElementById('ruleInsertAlert');
  if (alertElement) {
    alertElement.className = 'alert alert-info';
    alertElement.style.display = 'block';
    alertElement.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Inserting ${validRules.length} validated rules with retry logic...`;
  }

  // Process rules with improved batching and connection management
  let currentIndex = 0;
  const rulesToProcess = validRules;

  // Connection health tracking
  let consecutiveFailures = 0;
  const maxConsecutiveFailures = 3;

  function processNextRule() {
    if (currentIndex >= rulesToProcess.length) {
      // All rules processed, show final result
      if (alertElement) {
        if (successCount > 0 && errorCount === 0) {
          alertElement.className = 'alert alert-success';
          alertElement.innerHTML = `<i class="fas fa-check-circle"></i> Successfully inserted ${successCount} association rules!`;
        } else if (successCount > 0 && errorCount > 0) {
          alertElement.className = 'alert alert-warning';
          const errorSummary = errors.length > 3 ?
            `${errors.slice(0, 3).join(', ')} and ${errors.length - 3} more` :
            errors.join(', ');
          alertElement.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <strong>Partial Success:</strong> ${successCount} rules inserted successfully, ${errorCount} failed.
            <br><small><strong>Failed rules:</strong> ${errorSummary}</small>
          `;
        } else {
          alertElement.className = 'alert alert-danger';
          const errorSummary = errors.length > 5 ?
            `${errors.slice(0, 5).join(', ')} and ${errors.length - 5} more` :
            errors.join(', ');
          alertElement.innerHTML = `
            <i class="fas fa-times-circle"></i>
            <strong>Insertion Failed:</strong> All ${rulesToProcess.length} rules failed to insert.
            <br><small><strong>Failed rules:</strong> ${errorSummary}</small>
          `;
        }

        // Clear selection after successful insertion
        if (successCount > 0) {
          selectAllRules(false);
        }

        // Hide alert after 5 seconds
        setTimeout(() => {
          alertElement.style.display = 'none';
        }, 5000);
      }
      return;
    }

    const rule = rulesToProcess[currentIndex];

    // Enhanced insertion with connection health monitoring
    insertSingleRule(rule).then(() => {
      markRuleCardSaved(rule.card);
      successCount++;
      currentIndex++;
      consecutiveFailures = 0; // Reset failure counter on success

      // Update progress
      if (alertElement) {
        alertElement.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Inserting rules... (${currentIndex}/${rulesToProcess.length}) - ${successCount} successful`;
      }

      processNextRule();
    }).catch((error) => {
      consecutiveFailures++;

      // If too many consecutive failures, implement connection recovery delay
      if (consecutiveFailures >= maxConsecutiveFailures) {
        console.log(`Detected connection issues, implementing recovery delay...`);
        setTimeout(() => {
          consecutiveFailures = 0; // Reset after recovery
          // Retry this rule
          processNextRule();
        }, 2000); // 2 second recovery delay
        return;
      }

      errorCount++;
      const errorMsg = error.message || 'Unknown error';
      errors.push(`${rule.antecedent}→${rule.consequent} (${errorMsg})`);
      currentIndex++;
      processNextRule();
    });
  }

  processNextRule();
}

// Rule validation function
function validateRule(rule) {
  // Check if rule has valid antecedent and consequent
  if (!rule.antecedent || !rule.consequent) return false;

  // Check if antecedent and consequent are different
  if (rule.antecedent === rule.consequent) return false;

  // Check if support, confidence, and lift are valid numbers
  if (isNaN(rule.support) || isNaN(rule.confidence) || isNaN(rule.lift)) return false;

  // Check minimum thresholds
  if (rule.support <= 0 || rule.confidence <= 0 || rule.lift <= 0) return false;

  // Check maximum thresholds
  if (rule.support > 1 || rule.confidence > 1) return false;

  return true;
}

// Store for tracking inserted rules to prevent duplicates
let insertedRulesCache = new Set();

function insertSingleRule(rule, retryCount = 0) {
  return new Promise((resolve, reject) => {
    // Create unique identifier
    const ruleId = `${rule.antecedent}→${rule.consequent}`;

    // Check if rule was already inserted in this session
    if (insertedRulesCache.has(ruleId)) {
      resolve(); // Already inserted, consider it successful
      return;
    }

    // Prepare rule data for API
    const ruleData = {
      antecedent: Array.isArray(rule.antecedent) ? rule.antecedent : [rule.antecedent],
      consequent: Array.isArray(rule.consequent) ? rule.consequent : [rule.consequent],
      support: rule.support,
      confidence: rule.confidence,
      lift: rule.lift,
      rule_type: rule.rule_type || rule.type || rule.ruleType || 'product',
      min_support_threshold: rule.min_support_threshold || rule.support,
      min_confidence_threshold: rule.min_confidence_threshold,
      min_lift_threshold: rule.min_lift_threshold,
      source_view: rule.source_view || 'analysis.association_rules',
      metadata: rule.metadata || {
        antecedent_label: rule.antecedent,
        consequent_label: rule.consequent,
        source_path: window.location.pathname
      }
    };

    // Make actual API call to insert rule
    const maxRetries = 3;
    const baseDelay = 100;
    const backoffMultiplier = 1.5;

    function attemptInsertion() {
      const delay = baseDelay * Math.pow(backoffMultiplier, retryCount);

      setTimeout(() => {
        fetch('/analysis/api/association-rules/insert/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken()
          },
          body: JSON.stringify(ruleData)
        })
        .then(async (response) => {
          const result = await response.json().catch(() => ({ success: false, error: 'Invalid server response' }));

          if (response.ok && result.success) {
            // Successful insertion
            insertedRulesCache.add(ruleId);
            console.log(`Successfully inserted rule: ${ruleId}`);
            resolve();
          } else {
            throw new Error(result.error || 'Unknown server error');
          }
        })
        .catch((error) => {
          console.log(`Insertion attempt failed for ${ruleId}:`, error.message);

          if (retryCount < maxRetries) {
            // Retry with exponential backoff
            console.log(`Retrying insertion for ${ruleId} (attempt ${retryCount + 1}/${maxRetries})`);
            insertSingleRule(rule, retryCount + 1).then(resolve).catch(reject);
          } else {
            // Final failure after all retries
            reject(new Error(error.message || `Insertion failed after ${maxRetries} retries`));
          }
        });
      }, delay);
    }

    attemptInsertion();
  });
}

function initializeMultiSelect() {
  // Initialize the multi-select UI
  updateSelectedRulesCount();

  // Add CSS for selected rule cards
  const style = document.createElement('style');
  style.textContent = `
    .selectable-rule {
      transition: all 0.3s ease;
      position: relative;
    }
    .selectable-rule:has(.rule-checkbox:checked) {
      background: linear-gradient(135deg, #e8f5e8, #f0f8f0) !important;
      transform: translateX(2px);
    }
    .rule-checkbox {
      cursor: pointer;
    }
    .selectable-rule .rule-checkbox:checked {
      background-color: #28a745;
      border-color: #28a745;
    }
    .btn-group .btn.disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
  `;
  document.head.appendChild(style);
}
