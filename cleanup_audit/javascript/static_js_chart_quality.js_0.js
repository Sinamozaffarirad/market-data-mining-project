(function () {
  'use strict';

  const MIN_CHART_PIXEL_RATIO = 2;
  const MAX_CHART_PIXEL_RATIO = 3;
  let appliedPixelRatio = 0;
  let refreshTimer = null;

  window.chartRenderPixelRatio = function chartRenderPixelRatio() {
    const displayRatio = Number(window.devicePixelRatio || 1);
    return Math.min(
      MAX_CHART_PIXEL_RATIO,
      Math.max(MIN_CHART_PIXEL_RATIO, displayRatio * 1.5)
    );
  };

  function chartInstances() {
    if (!window.Chart || !Chart.instances) return [];
    if (Chart.instances instanceof Map) return Array.from(Chart.instances.values());
    return Object.values(Chart.instances);
  }

  function applyChartQuality(force) {
    if (!window.Chart) return false;
    const pixelRatio = window.chartRenderPixelRatio();
    Chart.defaults.devicePixelRatio = pixelRatio;
    Chart.defaults.responsive = true;
    if (!force && Math.abs(pixelRatio - appliedPixelRatio) < 0.01) return true;
    appliedPixelRatio = pixelRatio;

    chartInstances().forEach(chart => {
      if (!chart || !chart.canvas?.isConnected) return;
      if (chart.canvas.clientWidth < 1 || chart.canvas.clientHeight < 1) return;
      chart.options.devicePixelRatio = pixelRatio;
      chart.resize();
      chart.update('none');
    });
    return true;
  }

  function scheduleChartQualityRefresh(force) {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => applyChartQuality(!!force), 120);
  }

  window.refreshChartQuality = function refreshChartQuality() {
    scheduleChartQualityRefresh(true);
  };

  if (!applyChartQuality(true)) {
    document.addEventListener('DOMContentLoaded', () => applyChartQuality(true), {once: true});
    window.addEventListener('load', () => applyChartQuality(true), {once: true});
  }

  window.addEventListener('resize', () => scheduleChartQualityRefresh(false), {passive: true});
  window.visualViewport?.addEventListener('resize', () => scheduleChartQualityRefresh(false), {passive: true});
  document.addEventListener('shown.bs.modal', () => scheduleChartQualityRefresh(true));
  document.addEventListener('shown.bs.tab', () => scheduleChartQualityRefresh(true));
})();
